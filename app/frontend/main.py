import logging
import os
import time
import uuid
from datetime import datetime, timezone

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from starlette.requests import Request

logger = logging.getLogger("frontend_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
BACKEND_TIMEOUT_SECONDS = float(os.getenv("BACKEND_TIMEOUT_SECONDS", "30"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3000"))
DEFAULT_CORS_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]


def parse_cors_origins() -> list[str]:
    raw_value = os.getenv("CORS_ORIGINS")
    if not raw_value:
        return DEFAULT_CORS_ORIGINS
    origins = [origin.strip() for origin in raw_value.split(",") if origin.strip()]
    return origins or DEFAULT_CORS_ORIGINS


app = FastAPI(title="California Housing Frontend", version="1.0.0")
cors_origins = parse_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    logger.info("frontend req=%s start method=%s path=%s", request_id, request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("frontend req=%s error elapsed_ms=%s", request_id, elapsed_ms)
        raise
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info("frontend req=%s status=%s total_duration_ms=%s", request_id, response.status_code, elapsed_ms)
    return response

HTML_PAGE = rf"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Dự đoán giá nhà California</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 2rem auto; max-width: 980px; padding: 0 1.25rem; background: #f6f8fb; color: #1f2937; }}
        .card {{ background: white; border-radius: 18px; padding: 2rem; box-shadow: 0 12px 30px rgba(15,23,42,.08); }}
        h1 {{ margin-bottom: 0.5rem; }}
        .two-col {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 1rem; }}
        .field {{ display: flex; flex-direction: column; gap: 6px; }}
        label {{ font-size: 0.9rem; font-weight: 600; }}
        input, select {{ border: 1px solid #d1d5db; border-radius: 10px; padding: 0.7rem 0.8rem; font-size: 1rem; }}
        button {{ margin-top: 1rem; background: #2563eb; color: white; border: none; border-radius: 10px; padding: 0.9rem 1.4rem; cursor: pointer; font-weight: 600; }}
        .result {{ margin-top: 1.5rem; background: #ecfdf5; border: 1px solid #86efac; border-radius: 12px; padding: 1rem; }}
        .status {{ color: #475569; margin-top: 0.75rem; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Dự đoán giá nhà California</h1>
        <p>Nhập thông tin bất động sản để ước tính giá trị trung bình của căn nhà.</p>
        <form id="prediction-form">
            <div class="two-col" id="feature-fields"></div>
            <button id="submit-button" type="submit" disabled>Dự đoán</button>
        </form>
        <div class="status" id="status">Trạng thái: đang tải schema...</div>
        <div class="result" id="result" style="display:none;"></div>
    </div>

    <script>
        const apiBase = "{os.getenv("API_URL", "/")}".replace(/\/+$/, "");
        const form = document.getElementById('prediction-form');
        const fieldsContainer = document.getElementById('feature-fields');
        const submitButton = document.getElementById('submit-button');
        const resultBox = document.getElementById('result');
        const statusBox = document.getElementById('status');
        let featureSpecs = {{}};
        const featureLabels = {{
            longitude: 'Kinh độ',
            latitude: 'Vĩ độ',
            housing_median_age: 'Tuổi trung bình nhà',
            total_rooms: 'Tổng phòng',
            total_bedrooms: 'Tổng phòng ngủ',
            population: 'Dân số',
            households: 'Số hộ gia đình',
            median_income: 'Thu nhập trung bình',
            ocean_proximity: 'Vị trí gần biển'
        }};

        const getApiErrorMessage = (data) => {{
            if (!data) return 'Dự đoán thất bại';
            if (typeof data.detail === 'string') return data.detail;
            if (Array.isArray(data.detail)) {{
                const first = data.detail[0];
                if (first?.msg) return first.msg;
                return 'Lỗi xác thực';
            }}
            if (typeof data.detail === 'object') {{
                if (typeof data.detail.detail === 'string') return data.detail.detail;
                if (Array.isArray(data.detail.detail)) {{
                    const first = data.detail.detail[0];
                    if (first?.msg) return first.msg;
                }}
                return data.detail.error || 'Dự đoán thất bại';
            }}
            return 'Dự đoán thất bại';
        }};

        const makeField = (name, spec) => {{
            const wrapper = document.createElement('div');
            wrapper.className = 'field';
            if (spec.allowed_values) wrapper.style.gridColumn = '1 / -1';

            const id = `feature-${{name}}`;
            const label = document.createElement('label');
            label.htmlFor = id;
            label.textContent = featureLabels[name] || name;
            wrapper.appendChild(label);

            let input;
            if (spec.allowed_values) {{
                input = document.createElement('select');
                const placeholder = document.createElement('option');
                placeholder.value = '';
                placeholder.textContent = 'Chọn một tùy chọn';
                placeholder.disabled = !spec.nullable;
                placeholder.selected = true;
                input.appendChild(placeholder);
                for (const value of spec.allowed_values) {{
                    const option = document.createElement('option');
                    option.value = value;
                    option.textContent = value;
                    input.appendChild(option);
                }}
            }} else {{
                input = document.createElement('input');
                input.type = 'number';
                input.step = 'any';
                const limits = spec.range;
                const minimum = spec.min ?? (limits ? limits[0] : null);
                const maximum = spec.max ?? (limits ? limits[1] : null);
                if (minimum !== null && minimum !== undefined) input.min = minimum;
                if (maximum !== null && maximum !== undefined) input.max = maximum;
            }}

            input.id = id;
            input.name = name;
            input.required = !spec.nullable;
            wrapper.appendChild(input);
            return wrapper;
        }};

        const loadSchema = async () => {{
            const response = await fetch(`${{apiBase}}/api/schema`);
            const data = await response.json().catch(() => ({{}}));
            if (!response.ok) throw new Error(getApiErrorMessage(data));
            if (!data.input_features || typeof data.input_features !== 'object') {{
                throw new Error('Backend trả về schema không hợp lệ.');
            }}
            featureSpecs = data.input_features;
            fieldsContainer.replaceChildren(
                ...Object.entries(featureSpecs).map(([name, spec]) => makeField(name, spec))
            );
            submitButton.disabled = false;
            statusBox.textContent = 'Trạng thái: sẵn sàng';
        }};

        form.addEventListener('submit', async (event) => {{
            event.preventDefault();
            if (!form.reportValidity()) return;

            const features = {{}};
            for (const [name, spec] of Object.entries(featureSpecs)) {{
                const value = form.elements.namedItem(name).value;
                if (value === '' && spec.nullable) {{
                    features[name] = null;
                }} else if (spec.allowed_values) {{
                    features[name] = value;
                }} else {{
                    const numericValue = Number(value);
                    if (!Number.isFinite(numericValue)) {{
                        statusBox.textContent = `Trạng thái: giá trị ${{name}} phải là số hợp lệ`;
                        return;
                    }}
                    features[name] = numericValue;
                }}
            }}
            const payload = {{ features }};

            statusBox.textContent = 'Trạng thái: đang gửi yêu cầu...';
            try {{
                const response = await fetch(`${{apiBase}}/api/predict`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(payload)
                }});

                const data = await response.json().catch(() => ({{}}));
                if (!response.ok) {{
                    throw new Error(getApiErrorMessage(data));
                }}

                const prediction = Number(data.prediction).toLocaleString('en-US', {{ maximumFractionDigits: 2 }});
                resultBox.innerHTML = `
                    <strong>Kết quả dự đoán:</strong> ${{prediction}} USD<br>
                    <strong>Phiên bản mô hình:</strong> ${{data.model_version}}<br>
                    <strong>Mã yêu cầu:</strong> ${{data.request_id}}
                `;
                resultBox.style.display = 'block';
                statusBox.textContent = 'Trạng thái: thành công';
            }} catch (error) {{
                resultBox.textContent = `Lỗi: ${{error.message}}`;
                resultBox.style.display = 'block';
                statusBox.textContent = 'Trạng thái: thất bại';
            }}
        }});

        loadSchema().catch((error) => {{
            statusBox.textContent = `Không tải được schema: ${{error.message}}`;
        }});
    </script>
</body>
</html>
"""


@app.get("/health")
def health():
    return {"status": "ok", "service": "frontend", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def backend_proxy(path: str, request: Request):
    request_id = getattr(request.state, "request_id", request.headers.get("x-request-id") or uuid.uuid4().hex)
    body = await request.body()
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }
    headers["X-Request-ID"] = request_id
    try:
        response = requests.request(
            method=request.method,
            url=f"{BACKEND_URL}/api/{path}",
            params=request.query_params,
            headers=headers,
            data=body,
            timeout=BACKEND_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return Response(
            content='{"detail":{"error":"backend_unavailable","detail":"Backend service is unavailable."}}',
            status_code=502,
            media_type="application/json",
        )

    response_headers = {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in {"content-length", "transfer-encoding", "connection"}
    }
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=response_headers,
        media_type=response.headers.get("content-type"),
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=FRONTEND_PORT)
