//! Parsing and presentation for Tina's persistent TodoList commands.

use crate::conversation_memory::{TodoItem, TodoStatus};

pub(crate) const TODO_USAGE: &str = "Usage: /todo [list|add <task>|start <id>|done <id>|reopen <id>|remove <id>|clear-completed|help]";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TodoCommand {
    List,
    Add(String),
    Start(String),
    Done(String),
    Reopen(String),
    Remove(String),
    ClearCompleted,
    Help,
}

pub(crate) fn is_todo_command(command: &str) -> bool {
    let command = command.trim();
    command == "/todo"
        || command.starts_with("/todo ")
        || command == "/todos"
        || command.starts_with("/todos ")
        || command == "/待办"
        || command.starts_with("/待办 ")
        || command == "/任务"
        || command.starts_with("/任务 ")
}

pub(crate) fn parse_todo_command(command: &str) -> Result<TodoCommand, String> {
    let command = command.trim();
    let rest = command
        .strip_prefix("/todos")
        .or_else(|| command.strip_prefix("/todo"))
        .or_else(|| command.strip_prefix("/待办"))
        .or_else(|| command.strip_prefix("/任务"))
        .ok_or_else(|| TODO_USAGE.to_string())?
        .trim();
    if rest.is_empty() || rest == "list" {
        return Ok(TodoCommand::List);
    }
    if rest == "help" {
        return Ok(TodoCommand::Help);
    }
    if rest == "clear-completed" || rest == "clear completed" {
        return Ok(TodoCommand::ClearCompleted);
    }
    let (verb, value) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    let value = value.trim();
    let require_value = |label: &str| {
        if value.is_empty() {
            Err(format!("{label} requires a task or selector. {TODO_USAGE}"))
        } else {
            Ok(value.to_string())
        }
    };
    match verb {
        "add" => require_value("/todo add").map(TodoCommand::Add),
        "start" => require_value("/todo start").map(TodoCommand::Start),
        "done" => require_value("/todo done").map(TodoCommand::Done),
        "reopen" => require_value("/todo reopen").map(TodoCommand::Reopen),
        "remove" => require_value("/todo remove").map(TodoCommand::Remove),
        _ => Err(format!("Unknown TodoList action {verb:?}. {TODO_USAGE}")),
    }
}

#[allow(dead_code)]
pub(crate) fn render_todos(items: &[&TodoItem]) -> String {
    render_todos_localized(items, false)
}

pub(crate) fn render_todos_localized(items: &[&TodoItem], chinese: bool) -> String {
    if items.is_empty() {
        return if chinese {
            format!("待办为空。{TODO_USAGE}")
        } else {
            format!("TodoList is empty. {TODO_USAGE}")
        };
    }
    let mut output = if chinese {
        String::from("待办\n")
    } else {
        String::from("TodoList\n")
    };
    let mut pending = 0_usize;
    let mut in_progress = 0_usize;
    let mut completed = 0_usize;
    for (index, item) in items.iter().enumerate() {
        match item.status {
            TodoStatus::Pending => pending += 1,
            TodoStatus::InProgress => in_progress += 1,
            TodoStatus::Completed => completed += 1,
        }
        output.push_str(&format!(
            "{}. [{}] {} {}\n",
            index + 1,
            item.status.marker(),
            short_id(&item.todo_id),
            item.content
        ));
    }
    if chinese {
        output.push_str(&format!(
            "{} 待处理 | {} 进行中 | {} 已完成",
            pending, in_progress, completed
        ));
    } else {
        output.push_str(&format!(
            "{} pending | {} in progress | {} completed",
            pending, in_progress, completed
        ));
    }
    output
}

fn short_id(value: &str) -> &str {
    value.get(..12).unwrap_or(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: char, status: TodoStatus, content: &str) -> TodoItem {
        TodoItem {
            todo_id: id.to_string().repeat(64),
            content: content.to_string(),
            status,
            created_sequence: 1,
            updated_sequence: 1,
            removed: false,
        }
    }

    #[test]
    fn parser_is_explicit_and_preserves_task_text() {
        assert_eq!(parse_todo_command("/todo").unwrap(), TodoCommand::List);
        assert_eq!(
            parse_todo_command("/todos list").unwrap(),
            TodoCommand::List
        );
        assert_eq!(parse_todo_command("/待办 list").unwrap(), TodoCommand::List);
        assert_eq!(
            parse_todo_command("/todo add  inspect KV owner  ").unwrap(),
            TodoCommand::Add("inspect KV owner".to_string())
        );
        assert_eq!(
            parse_todo_command("/todo done 2").unwrap(),
            TodoCommand::Done("2".to_string())
        );
        assert_eq!(
            parse_todo_command("/todo clear completed").unwrap(),
            TodoCommand::ClearCompleted
        );
        assert!(parse_todo_command("/todo remove").is_err());
        assert!(parse_todo_command("/plugin todo").is_err());
    }

    #[test]
    fn rendering_uses_stable_indices_ids_and_status_counts() {
        let first = item('a', TodoStatus::Pending, "audit cache");
        let second = item('b', TodoStatus::InProgress, "implement handoff");
        let third = item('c', TodoStatus::Completed, "write tests");
        let rendered = render_todos(&[&first, &second, &third]);
        assert!(rendered.contains("1. [ ] aaaaaaaaaaaa audit cache"));
        assert!(rendered.contains("2. [>] bbbbbbbbbbbb implement handoff"));
        assert!(rendered.contains("3. [x] cccccccccccc write tests"));
        assert!(rendered.contains("1 pending | 1 in progress | 1 completed"));
    }
}
