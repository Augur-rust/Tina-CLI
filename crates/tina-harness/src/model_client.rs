//! Cancellable Chat Completions transport with bounded incremental SSE parsing.
use crate::live_config::LiveConfig;
#[cfg(test)]
use serde_json::json;
use serde_json::Value;
use std::{collections::BTreeMap, time::Duration};
const MAX_RESPONSE: usize = 32 * 1024 * 1024;
const MAX_FRAME: usize = 1024 * 1024;
#[derive(Clone, Debug, Default)]
pub(crate) struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}
#[derive(Debug, Default)]
pub(crate) struct Completion {
    pub content: String,
    pub reasoning: String,
    pub calls: Vec<ToolCall>,
    pub finish: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}
#[derive(Clone, Copy)]
pub(crate) enum PieceKind {
    Content,
    Reasoning,
}
pub(crate) struct ModelClient {
    runtime: tokio::runtime::Runtime,
    client: reqwest::Client,
    config: LiveConfig,
    key: String,
}
impl ModelClient {
    pub fn new(config: LiveConfig) -> Result<Self, String> {
        config.validate()?;
        let key = config.credential()?;
        let url = reqwest::Url::parse(&config.base_url).map_err(|e| e.to_string())?;
        let mut builder = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(config.timeout_seconds))
            .redirect(reqwest::redirect::Policy::none());
        if matches!(
            url.host_str(),
            Some("localhost" | "127.0.0.1" | "[::1]" | "::1")
        ) {
            builder = builder.no_proxy();
        }
        let client = builder.build().map_err(|e| e.to_string())?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self {
            runtime,
            client,
            config,
            key,
        })
    }
    pub fn redact(&self, message: &str) -> String {
        if self.key.is_empty() {
            message.into()
        } else {
            message.replace(&self.key, "[redacted]")
        }
    }
    pub fn stream(
        &self,
        payload: &Value,
        on_piece: &mut impl FnMut(PieceKind, &str) -> Result<(), String>,
        poll_cancel: &mut impl FnMut() -> Result<bool, String>,
    ) -> Result<Completion, String> {
        let result=self.runtime.block_on(async {
            let work=async {
                let mut request=self.client.post(self.config.endpoint("chat/completions")).json(payload);
                if !self.key.is_empty(){request=request.bearer_auth(&self.key);}
                let mut response=request.send().await.map_err(|e|e.without_url().to_string())?;
                if !response.status().is_success(){
                    let status=response.status();let mut bytes=Vec::new();
                    while let Some(chunk)=response.chunk().await.map_err(|e|e.without_url().to_string())?{let n=chunk.len().min(8192usize.saturating_sub(bytes.len()));bytes.extend_from_slice(&chunk[..n]);if bytes.len()>=8192{break;}}
                    return Err(format!("HTTP {status}: {}",String::from_utf8_lossy(&bytes)));
                }
                let sse=response.headers().get(reqwest::header::CONTENT_TYPE).and_then(|v|v.to_str().ok()).is_some_and(|v|v.starts_with("text/event-stream"));
                let mut parser=StreamParser::default();let mut plain=Vec::new();let mut total=0usize;
                while let Some(chunk)=response.chunk().await.map_err(|e|e.without_url().to_string())?{
                    total=total.saturating_add(chunk.len());if total>MAX_RESPONSE{return Err("Model response exceeds 32 MiB".into());}
                    if sse {parser.feed(&chunk,on_piece)?;if parser.done{break;}} else {plain.extend_from_slice(&chunk);}
                }
                if sse{parser.finish(on_piece)}else{
                    let value:Value=serde_json::from_slice(&plain).map_err(|e|format!("Invalid model JSON: {e}"))?;
                    parser.consume(&value,on_piece)?;parser.finish(on_piece)
                }
            };
            tokio::pin!(work);let mut tick=tokio::time::interval(Duration::from_millis(40));
            loop{tokio::select!{
                result=&mut work=>return result,
                _=tick.tick()=>if poll_cancel()?{return Err("Generation cancelled; request disconnected. Partial output was not committed.".into());}
            }}
        });
        result.map_err(|e| self.redact(&e))
    }
}
#[derive(Default)]
struct StreamParser {
    buffer: Vec<u8>,
    result: Completion,
    calls: BTreeMap<usize, ToolCall>,
    done: bool,
}
impl StreamParser {
    fn feed(
        &mut self,
        bytes: &[u8],
        on_piece: &mut impl FnMut(PieceKind, &str) -> Result<(), String>,
    ) -> Result<(), String> {
        self.buffer.extend_from_slice(bytes);
        loop {
            let lf = self
                .buffer
                .windows(2)
                .position(|w| w == b"\n\n")
                .map(|p| (p, 2));
            let crlf = self
                .buffer
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|p| (p, 4));
            let split = lf.into_iter().chain(crlf).min_by_key(|pair| pair.0);
            let Some((end, sep)) = split else {
                break;
            };
            if end > MAX_FRAME {
                return Err("SSE frame exceeds 1 MiB".into());
            }
            let frame = self.buffer[..end].to_vec();
            self.buffer.drain(..end + sep);
            self.frame(&frame, on_piece)?;
            if self.done {
                break;
            }
        }
        if self.buffer.len() > MAX_FRAME {
            return Err("SSE frame exceeds 1 MiB".into());
        }
        Ok(())
    }
    fn frame(
        &mut self,
        frame: &[u8],
        on_piece: &mut impl FnMut(PieceKind, &str) -> Result<(), String>,
    ) -> Result<(), String> {
        let text = std::str::from_utf8(frame).map_err(|_| "Invalid UTF-8 in SSE")?;
        let data = text
            .lines()
            .filter_map(|s| {
                s.strip_prefix("data:")
                    .map(|s| s.strip_prefix(' ').unwrap_or(s))
            })
            .collect::<Vec<_>>()
            .join("\n");
        if data.is_empty() {
            return Ok(());
        }
        if data.trim() == "[DONE]" {
            self.done = true;
            return Ok(());
        }
        let value: Value =
            serde_json::from_str(&data).map_err(|e| format!("Invalid SSE JSON: {e}"))?;
        self.consume(&value, on_piece)
    }
    fn consume(
        &mut self,
        v: &Value,
        on_piece: &mut impl FnMut(PieceKind, &str) -> Result<(), String>,
    ) -> Result<(), String> {
        if let Some(e) = v.get("error") {
            return Err(format!("Model error: {e}"));
        }
        if let Some(usage) = v.get("usage").filter(|u| !u.is_null()) {
            self.result.prompt_tokens = usage["prompt_tokens"]
                .as_u64()
                .unwrap_or(self.result.prompt_tokens as u64)
                as usize;
            self.result.completion_tokens = usage["completion_tokens"]
                .as_u64()
                .unwrap_or(self.result.completion_tokens as u64)
                as usize;
        }
        let Some(choices) = v["choices"].as_array() else {
            return Ok(());
        };
        for c in choices {
            if c["index"].as_u64().unwrap_or(0) != 0 {
                continue;
            }
            if let Some(reason) = c["finish_reason"].as_str() {
                self.result.finish = reason.into();
            }
            let d = c
                .get("delta")
                .or_else(|| c.get("message"))
                .unwrap_or(&Value::Null);
            for (field, kind) in [
                ("content", PieceKind::Content),
                ("reasoning_content", PieceKind::Reasoning),
                ("reasoning", PieceKind::Reasoning),
            ] {
                if let Some(s) = d[field].as_str() {
                    match kind {
                        PieceKind::Content => self.result.content.push_str(s),
                        PieceKind::Reasoning => self.result.reasoning.push_str(s),
                    };
                    on_piece(kind, s)?;
                }
            }
            if let Some(calls) = d["tool_calls"].as_array() {
                for (ordinal, c) in calls.iter().enumerate() {
                    let index = c["index"].as_u64().unwrap_or(ordinal as u64) as usize;
                    if index >= 32 {
                        return Err("Model requested more than 32 tools in one batch".into());
                    }
                    let call = self.calls.entry(index).or_default();
                    if let Some(id) = c["id"].as_str() {
                        call.id.push_str(id);
                    }
                    if let Some(name) = c["function"]["name"].as_str() {
                        call.name.push_str(name);
                    }
                    if let Some(args) = c["function"]["arguments"].as_str() {
                        call.arguments.push_str(args);
                    }
                    if call.arguments.len() > 256 * 1024
                        || call.name.len() > 256
                        || call.id.len() > 128
                    {
                        return Err("Tool call exceeded its bounds".into());
                    }
                }
            }
        }
        Ok(())
    }
    fn finish(
        mut self,
        on_piece: &mut impl FnMut(PieceKind, &str) -> Result<(), String>,
    ) -> Result<Completion, String> {
        if !self.buffer.is_empty() && !self.done {
            let frame = std::mem::take(&mut self.buffer);
            self.frame(&frame, on_piece)?;
        }
        if !["stop", "tool_calls"].contains(&self.result.finish.as_str()) {
            return Err(format!("Model response incomplete (finish_reason={:?}); no partial answer or tools committed. Increase --max-tokens for a length limit.",self.result.finish));
        }
        self.result.calls = self.calls.into_values().collect();
        if self.result.finish == "tool_calls" && self.result.calls.is_empty() {
            return Err("Model reported tool_calls without a call".into());
        }
        if !self.result.calls.is_empty() && self.result.finish != "tool_calls" {
            return Err("Tool calls without a tool_calls finish marker".into());
        }
        for c in &self.result.calls {
            if c.id.is_empty() || c.name.is_empty() {
                return Err("Incomplete tool call identity".into());
            }
            tina_protocol::validate_strict_json(c.arguments.as_bytes())
                .map_err(|e| format!("Invalid tool arguments: {e}"))?;
            if !serde_json::from_str::<Value>(&c.arguments)
                .map_err(|e| e.to_string())?
                .is_object()
            {
                return Err("Tool arguments must be an object".into());
            }
        }
        if self.result.calls.is_empty() && self.result.content.trim().is_empty() {
            return Err("Model returned no final answer".into());
        }
        Ok(self.result)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sse_utf8_and_tool_arguments_can_split_at_every_byte() {
        let v = json!({"choices":[{"index":0,"delta":{"content":"你好","tool_calls":[{"index":0,"id":"c1","function":{"name":"read","arguments":"{\"path\":"}}]}}]});
        let w = json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":42,"completion_tokens":12}});
        let text = format!("data: {v}\r\n\r\ndata: {w}\n\ndata: [DONE]\n\n");
        let mut p = StreamParser::default();
        let mut got = String::new();
        let mut cb = |_, s: &str| {
            got.push_str(s);
            Ok(())
        };
        for b in text.as_bytes() {
            p.feed(&[*b], &mut cb).unwrap();
        }
        let r = p.finish(&mut cb).unwrap();
        assert_eq!(got, "你好");
        assert_eq!(r.calls[0].arguments, "{\"path\":\"x\"}");
        assert_eq!(r.prompt_tokens, 42);
    }
    #[test]
    fn truncation_never_authorizes_a_tool() {
        let mut p = StreamParser::default();
        p.consume(&json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"1","function":{"name":"delete","arguments":"{}"}}]},"finish_reason":"length"}]}),&mut |_,_|Ok(())).unwrap();
        assert!(p.finish(&mut |_, _| Ok(())).is_err());
    }
    #[test]
    fn malformed_or_duplicate_tool_arguments_rejected() {
        let mut p = StreamParser::default();
        p.consume(&json!({"choices":[{"message":{"tool_calls":[{"id":"1","function":{"name":"read","arguments":"{\"path\":\"a\",\"path\":\"b\"}"}}]},"finish_reason":"tool_calls"}]}),&mut |_,_|Ok(())).unwrap();
        assert!(p.finish(&mut |_, _| Ok(())).is_err());
    }
    #[test]
    fn mixed_sse_delimiters_in_one_chunk_keep_event_order() {
        let first = json!({"choices":[{"delta":{"content":"first "}}]});
        let second = json!({"choices":[{"delta":{"content":"second"},"finish_reason":"stop"}]});
        let input = format!("data: {first}\r\n\r\ndata: {second}\n\ndata: [DONE]\n\n");
        let mut p = StreamParser::default();
        p.feed(input.as_bytes(), &mut |_, _| Ok(())).unwrap();
        assert_eq!(
            p.finish(&mut |_, _| Ok(())).unwrap().content,
            "first second"
        );
    }
    #[test]
    fn cancellation_closes_socket_before_first_token() {
        use std::{
            io::{Read, Write},
            net::TcpListener,
            time::Instant,
        };
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut data = Vec::new();
            let mut b = [0; 1024];
            loop {
                let n = socket.read(&mut b).unwrap();
                assert!(n > 0);
                data.extend_from_slice(&b[..n]);
                if let Some(end) = data.windows(4).position(|x| x == b"\r\n\r\n") {
                    let header = String::from_utf8_lossy(&data[..end]).to_ascii_lowercase();
                    let length = header
                        .lines()
                        .find_map(|s| s.strip_prefix("content-length:"))
                        .unwrap()
                        .trim()
                        .parse::<usize>()
                        .unwrap();
                    if data.len() >= end + 4 + length {
                        break;
                    }
                }
            }
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").unwrap();
            sender.send(()).unwrap();
            let closed = match socket.read(&mut b) {
                Ok(0) => true,
                Err(e) => e.kind() == std::io::ErrorKind::ConnectionReset,
                _ => false,
            };
            assert!(closed, "cancelled stream left the server socket open");
        });
        let c = LiveConfig {
            base_url: format!("http://127.0.0.1:{port}/v1"),
            api_key_env: String::new(),
            ..Default::default()
        };
        let client = ModelClient::new(c).unwrap();
        let started = Instant::now();
        let result = client.stream(&json!({"stream":true}), &mut |_, _| Ok(()), &mut || {
            Ok(receiver.try_recv().is_ok())
        });
        assert!(result.unwrap_err().contains("cancelled"));
        assert!(started.elapsed() < Duration::from_secs(2));
        worker.join().unwrap();
    }
}
