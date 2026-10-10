use std::collections::BTreeMap;

use crate::CheckResult;

pub enum Value {
    Null,
    Bool,
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

impl Value {
    pub fn object(&self) -> CheckResult<&BTreeMap<String, Value>> {
        match self {
            Self::Object(value) => Ok(value),
            _ => Err("expected a JSON object".to_owned()),
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Object(value) => value.get(key),
            _ => None,
        }
    }

    pub fn string(&self) -> CheckResult<&str> {
        match self {
            Self::String(value) => Ok(value),
            _ => Err("expected a JSON string".to_owned()),
        }
    }

    pub fn number(&self) -> CheckResult<f64> {
        match self {
            Self::Number(value) => Ok(*value),
            _ => Err("expected a JSON number".to_owned()),
        }
    }
}

pub fn parse(text: &str) -> CheckResult<Value> {
    let mut parser = Parser { text, position: 0 };
    let value = parser.value(0)?;
    parser.whitespace();
    if parser.position != text.len() {
        return Err(parser.error("trailing JSON data"));
    }
    Ok(value)
}

struct Parser<'a> {
    text: &'a str,
    position: usize,
}

impl Parser<'_> {
    fn error(&self, message: &str) -> String {
        format!("JSON byte {}: {message}", self.position)
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.position).copied()
    }

    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.position += 1;
        }
    }

    fn take(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, byte: u8) -> CheckResult<()> {
        self.whitespace();
        if self.take(byte) {
            Ok(())
        } else {
            Err(self.error(&format!("expected {:?}", char::from(byte))))
        }
    }

    fn literal(&mut self, literal: &str) -> CheckResult<()> {
        if self.text[self.position..].starts_with(literal) {
            self.position += literal.len();
            Ok(())
        } else {
            Err(self.error("invalid literal"))
        }
    }

    fn value(&mut self, depth: usize) -> CheckResult<Value> {
        if depth > 128 {
            return Err(self.error("nesting limit exceeded"));
        }
        self.whitespace();
        match self.peek() {
            Some(b'{') => {
                self.position += 1;
                self.whitespace();
                let mut object = BTreeMap::new();
                if self.take(b'}') {
                    return Ok(Value::Object(object));
                }
                loop {
                    self.whitespace();
                    let key = self.string()?;
                    self.expect(b':')?;
                    let value = self.value(depth + 1)?;
                    if object.insert(key, value).is_some() {
                        return Err(self.error("duplicate object key"));
                    }
                    self.whitespace();
                    if self.take(b'}') {
                        break;
                    }
                    self.expect(b',')?;
                }
                Ok(Value::Object(object))
            }
            Some(b'[') => {
                self.position += 1;
                self.whitespace();
                let mut array = Vec::new();
                if self.take(b']') {
                    return Ok(Value::Array(array));
                }
                loop {
                    array.push(self.value(depth + 1)?);
                    self.whitespace();
                    if self.take(b']') {
                        break;
                    }
                    self.expect(b',')?;
                }
                Ok(Value::Array(array))
            }
            Some(b'"') => self.string().map(Value::String),
            Some(b'n') => {
                self.literal("null")?;
                Ok(Value::Null)
            }
            Some(b't') => {
                self.literal("true")?;
                Ok(Value::Bool)
            }
            Some(b'f') => {
                self.literal("false")?;
                Ok(Value::Bool)
            }
            Some(b'-' | b'0'..=b'9') => self.number().map(Value::Number),
            _ => Err(self.error("expected a value")),
        }
    }

    fn digits(&mut self) -> CheckResult<()> {
        let start = self.position;
        while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
            self.position += 1;
        }
        if start == self.position {
            Err(self.error("expected digits"))
        } else {
            Ok(())
        }
    }

    fn number(&mut self) -> CheckResult<f64> {
        let start = self.position;
        self.take(b'-');
        match self.peek() {
            Some(b'0') => self.position += 1,
            Some(b'1'..=b'9') => self.digits()?,
            _ => return Err(self.error("invalid number")),
        }
        if self.take(b'.') {
            self.digits()?;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.position += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.position += 1;
            }
            self.digits()?;
        }
        let value: f64 = self.text[start..self.position]
            .parse()
            .map_err(|_| self.error("invalid number"))?;
        if !value.is_finite() {
            return Err(self.error("number is outside the finite f64 range"));
        }
        Ok(value)
    }

    fn hex4(&mut self) -> CheckResult<u32> {
        let mut value = 0u32;
        for _ in 0..4 {
            let byte = self
                .peek()
                .ok_or_else(|| self.error("truncated Unicode escape"))?;
            let digit = match byte {
                b'0'..=b'9' => u32::from(byte - b'0'),
                b'a'..=b'f' => u32::from(byte - b'a' + 10),
                b'A'..=b'F' => u32::from(byte - b'A' + 10),
                _ => return Err(self.error("invalid Unicode escape")),
            };
            self.position += 1;
            value = value * 16 + digit;
        }
        Ok(value)
    }

    fn string(&mut self) -> CheckResult<String> {
        if !self.take(b'"') {
            return Err(self.error("expected a string"));
        }
        let mut output = String::new();
        loop {
            let start = self.position;
            while self
                .peek()
                .is_some_and(|byte| byte >= 0x20 && byte != b'"' && byte != b'\\')
            {
                self.position += 1;
            }
            output.push_str(&self.text[start..self.position]);
            match self.peek() {
                Some(b'"') => {
                    self.position += 1;
                    return Ok(output);
                }
                Some(b'\\') => self.position += 1,
                Some(_) => return Err(self.error("unescaped control character")),
                None => return Err(self.error("unterminated string")),
            }
            let escape = self
                .peek()
                .ok_or_else(|| self.error("truncated escape"))?;
            self.position += 1;
            match escape {
                b'"' => output.push('"'),
                b'\\' => output.push('\\'),
                b'/' => output.push('/'),
                b'b' => output.push('\u{0008}'),
                b'f' => output.push('\u{000c}'),
                b'n' => output.push('\n'),
                b'r' => output.push('\r'),
                b't' => output.push('\t'),
                b'u' => {
                    let mut code = self.hex4()?;
                    if (0xd800..=0xdbff).contains(&code) {
                        if !self.take(b'\\') || !self.take(b'u') {
                            return Err(self.error("missing low surrogate"));
                        }
                        let low = self.hex4()?;
                        if !(0xdc00..=0xdfff).contains(&low) {
                            return Err(self.error("invalid low surrogate"));
                        }
                        code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
                    }
                    let character = char::from_u32(code)
                        .ok_or_else(|| self.error("invalid Unicode scalar"))?;
                    output.push(character);
                }
                _ => return Err(self.error("invalid escape")),
            }
        }
    }
}
