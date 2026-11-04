//! Errors as OpenAI-shaped JSON. Messages never contain file paths: runtime errors that
//! may (a model file that cannot be read, say) are logged with the request id and the
//! client gets a generic message naming that id.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use qubivex_runtime::RuntimeError;
use serde_json::json;

/// An error returned to the client.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiError {
    /// HTTP status.
    pub status: StatusCode,
    /// OpenAI error `type`.
    pub kind: &'static str,
    /// OpenAI error `code`.
    pub code: &'static str,
    /// Human-readable message (no paths).
    pub message: String,
    /// The parameter at fault, if any.
    pub param: Option<&'static str>,
}

impl ApiError {
    fn new(status: StatusCode, kind: &'static str, code: &'static str, message: String) -> Self {
        Self {
            status,
            kind,
            code,
            message,
            param: None,
        }
    }

    /// 400: the request is malformed or asks for something unsupported.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_request",
            message.into(),
        )
    }

    /// 400 naming the parameter at fault.
    pub fn invalid_param(param: &'static str, message: impl Into<String>) -> Self {
        Self {
            param: Some(param),
            ..Self::invalid(message)
        }
    }

    /// 401: a missing or wrong key.
    pub fn unauthorized() -> Self {
        Self::new(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid_api_key",
            "missing or invalid bearer token".into(),
        )
    }

    /// 403: the admin endpoints are disabled (no admin token configured).
    pub fn admin_disabled() -> Self {
        Self::new(
            StatusCode::FORBIDDEN,
            "permission_error",
            "admin_disabled",
            "the admin endpoints are disabled; start the server with an admin token".into(),
        )
    }

    /// 404: no such model is loaded.
    pub fn model_not_found(id: &str) -> Self {
        Self {
            param: Some("model"),
            ..Self::new(
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                "model_not_found",
                format!("the model {id:?} is not loaded"),
            )
        }
    }

    /// 404 for an unknown route.
    pub fn not_found() -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "not_found",
            "no such endpoint".into(),
        )
    }

    /// 409: a model with this id is already loaded.
    pub fn conflict(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "conflict",
            message.into(),
        )
    }

    /// 429: every generation slot of the model is busy for longer than the queue wait.
    pub fn busy(slots: usize) -> Self {
        Self::new(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "too_many_requests",
            format!("all {slots} generation slots are busy; retry later"),
        )
    }

    /// 503: memory or storage cannot take another request now.
    pub fn overloaded(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "overloaded",
            message.into(),
        )
    }

    /// 500 with a message that names only the request id; the detail goes to the log.
    pub fn internal(request_id: &str) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "internal_error",
            format!(
                "the request failed; the server log has the details under request id {request_id}"
            ),
        )
    }

    /// A runtime error as the client should see it. Errors whose text can carry paths
    /// or internals are logged and replaced by [`ApiError::internal`].
    pub fn from_runtime(e: &RuntimeError, request_id: &str) -> Self {
        match e {
            // Memory and storage refusals are about capacity, and their text is built
            // from sizes and counts only.
            RuntimeError::Memory(m) => Self::overloaded(format!("memory: {m}")),
            RuntimeError::Storage(m) => Self::overloaded(format!("storage: {m}")),
            other => {
                eprintln!("qubivex-api: request {request_id}: {other}");
                Self::internal(request_id)
            }
        }
    }

    /// The OpenAI error body.
    #[must_use]
    pub fn body(&self) -> serde_json::Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": self.param,
                "code": self.code,
            }
        })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
    }
}
