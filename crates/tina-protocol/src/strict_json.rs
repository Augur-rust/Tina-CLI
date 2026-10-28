use std::{collections::BTreeMap, fmt};

use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};

use crate::ProfileError;

#[derive(Clone, Debug)]
pub(crate) enum StrictValue {
    Null,
    Bool,
    Number(serde_json::Number),
    String(String),
    Array,
    Object(BTreeMap<String, Self>),
}

impl StrictValue {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, ProfileError> {
        let mut parser = serde_json::Deserializer::from_slice(bytes);
        let value = Self::deserialize(&mut parser)
            .map_err(|error| ProfileError::Json(error.to_string()))?;
        parser
            .end()
            .map_err(|error| ProfileError::Json(error.to_string()))?;
        Ok(value)
    }
    pub(crate) fn object(
        &self,
        field: &'static str,
    ) -> Result<&BTreeMap<String, Self>, ProfileError> {
        match self {
            Self::Object(value) => Ok(value),
            _ => Err(ProfileError::InvalidField(field)),
        }
    }
    pub(crate) fn string(&self, field: &'static str) -> Result<&str, ProfileError> {
        match self {
            Self::String(value) => Ok(value),
            _ => Err(ProfileError::InvalidField(field)),
        }
    }
    pub(crate) fn usize(&self, field: &'static str) -> Result<usize, ProfileError> {
        match self {
            Self::Number(value) => value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or(ProfileError::InvalidField(field)),
            _ => Err(ProfileError::InvalidField(field)),
        }
    }
    pub(crate) fn f32(&self, field: &'static str) -> Result<f32, ProfileError> {
        let value = match self {
            Self::Number(value) => value.as_f64(),
            _ => None,
        }
        .ok_or(ProfileError::InvalidField(field))? as f32;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(ProfileError::InvalidField(field))
        }
    }
}

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StrictVisitor;
        impl<'de> Visitor<'de> for StrictVisitor {
            type Value = StrictValue;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("JSON without duplicate keys")
            }
            fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
                Ok(StrictValue::Bool)
            }
            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(StrictValue::Number(value.into()))
            }
            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
                Ok(StrictValue::Number(value.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(StrictValue::Number)
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }
            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
                Ok(StrictValue::String(value.into()))
            }
            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(StrictValue::String(value))
            }
            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(StrictValue::Null)
            }
            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(StrictValue::Null)
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                while sequence.next_element::<StrictValue>()?.is_some() {}
                Ok(StrictValue::Array)
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, StrictValue>()? {
                    if values.insert(key.clone(), value).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate JSON key: {key}"
                        )));
                    }
                }
                Ok(StrictValue::Object(values))
            }
        }
        deserializer.deserialize_any(StrictVisitor)
    }
}

pub(crate) fn required<'a>(
    object: &'a BTreeMap<String, StrictValue>,
    field: &'static str,
) -> Result<&'a StrictValue, ProfileError> {
    object.get(field).ok_or(ProfileError::MissingField(field))
}
