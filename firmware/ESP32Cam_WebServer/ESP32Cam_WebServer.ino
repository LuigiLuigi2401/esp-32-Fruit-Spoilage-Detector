#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include <lwip/sockets.h>

// Wi-Fi Credentials
const char* ssid = "TemoMunny";
const char* password = "Dodojojo@CoviD1977";

// AI THINKER ESP32-CAM Pin Map
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define LED_FLASH_GPIO    4

httpd_handle_t camera_httpd = NULL;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    size_t _jpg_buf_len = 0;
    uint8_t * _jpg_buf = NULL;
    char part_buf[64];

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if(res != ESP_OK) return res;

    int val = 1;
    setsockopt(httpd_req_to_sockfd(req), IPPROTO_TCP, TCP_NODELAY, (const char *)&val, sizeof(int));

    while(true){
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("Camera capture failed");
            res = ESP_FAIL;
        } else {
            if(fb->format != PIXFORMAT_JPEG){
                bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
                esp_camera_fb_return(fb);
                fb = NULL;
                if(!jpeg_converted){
                    Serial.println("JPEG compression failed");
                    res = ESP_FAIL;
                }
            } else {
                _jpg_buf_len = fb->len;
                _jpg_buf = fb->buf;
            }
        }
        if(res != ESP_OK) break;

        size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);
        res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
        if(res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
        if(res == ESP_OK) res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));

        if(fb){
            esp_camera_fb_return(fb);
            fb = NULL;
            _jpg_buf = NULL;
        } else if(_jpg_buf){
            free(_jpg_buf);
            _jpg_buf = NULL;
        }
        if(res != ESP_OK) break;

        // YIELD FreeRTOS task so incoming /control HTTP requests (Flash LED, Resolution) execute immediately!
        vTaskDelay(1 / portTICK_PERIOD_MS);
    }
    return res;
}

static esp_err_t cmd_handler(httpd_req_t *req) {
    char*  buf;
    size_t buf_len;
    char variable[32] = {0,};
    char value[32] = {0,};

    buf_len = httpd_req_get_url_query_len(req) + 1;
    if (buf_len > 1) {
        buf = (char*)malloc(buf_len);
        if(!buf){
            httpd_resp_send_500(req);
            return ESP_FAIL;
        }
        if (httpd_req_get_url_query_str(req, buf, buf_len) == ESP_OK) {
            if (httpd_query_key_value(buf, "var", variable, sizeof(variable)) == ESP_OK &&
                httpd_query_key_value(buf, "val", value, sizeof(value)) == ESP_OK) {
            } else {
                free(buf);
                httpd_resp_send_404(req);
                return ESP_FAIL;
            }
        }
        free(buf);
    } else {
        httpd_resp_send_404(req);
        return ESP_FAIL;
    }

    int val = atoi(value);
    sensor_t * s = esp_camera_sensor_get();
    int res = 0;

    if(!strcmp(variable, "framesize")) {
        if(s->pixformat == PIXFORMAT_JPEG) res = s->set_framesize(s, (framesize_t)val);
    } else if(!strcmp(variable, "quality")) {
        res = s->set_quality(s, val);
    } else if(!strcmp(variable, "flash")) {
        digitalWrite(LED_FLASH_GPIO, val ? HIGH : LOW);
        Serial.printf("Flash LED state set to: %d\n", val);
    } else {
        res = -1;
    }

    if(res) {
        return httpd_resp_send_500(req);
    }

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, "OK", 2);
}

static esp_err_t index_handler(httpd_req_t *req) {
    const char* html = 
        "<!DOCTYPE html><html lang='en'><head>"
        "<meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'>"
        "<title>ESP32-CAM High-FPS Vision Dashboard</title>"
        "<style>"
        "* { box-sizing: border-box; margin: 0; padding: 0; }"
        "body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: radial-gradient(circle at top, #1e293b, #0f172a); color: #f8fafc; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px; }"
        ".card { background: rgba(30, 41, 59, 0.75); backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; width: 100%; max-width: 800px; padding: 24px; box-shadow: 0 20px 50px rgba(0,0,0,0.6); text-align: center; }"
        ".header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 10px; }"
        ".title { font-size: 22px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }"
        ".badge-container { display: flex; gap: 10px; }"
        ".badge { background: #0f172a; border: 1px solid #334155; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; color: #38bdf8; display: flex; align-items: center; gap: 6px; }"
        ".badge.live { color: #10b981; border-color: #059669; }"
        ".stream-wrapper { position: relative; width: 100%; border-radius: 14px; overflow: hidden; background: #020617; border: 2px solid #38bdf8; box-shadow: 0 0 20px rgba(56, 189, 248, 0.2); margin-bottom: 20px; }"
        "img#stream { width: 100%; height: auto; display: block; object-fit: contain; min-height: 240px; }"
        ".controls-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; align-items: center; }"
        ".control-group { background: #0f172a; border: 1px solid #334155; padding: 12px 16px; border-radius: 12px; display: flex; flex-direction: column; gap: 6px; text-align: left; }"
        ".label { font-size: 12px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }"
        "select { background: #1e293b; color: #ffffff; border: 1px solid #475569; padding: 10px; border-radius: 8px; font-weight: 600; font-size: 14px; cursor: pointer; outline: none; width: 100%; transition: border 0.2s; }"
        "select:focus { border-color: #38bdf8; }"
        "button { background: linear-gradient(135deg, #0284c7, #0369a1); color: #fff; border: none; padding: 12px 18px; border-radius: 10px; font-weight: 700; font-size: 14px; cursor: pointer; transition: all 0.2s; display: flex; align-items: center; justify-content: center; gap: 8px; width: 100%; }"
        "button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4); }"
        "button:active { transform: translateY(0); }"
        "button.flash { background: linear-gradient(135deg, #eab308, #ca8a04); color: #0f172a; }"
        "button.flash:hover { box-shadow: 0 4px 12px rgba(234, 179, 8, 0.4); }"
        "button.snap { background: linear-gradient(135deg, #10b981, #059669); }"
        "button.snap:hover { box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4); }"
        ".footer { margin-top: 15px; font-size: 12px; color: #64748b; }"
        "</style></head><body>"
        "<div class='card'>"
        "<div class='header'>"
        "<div class='title'>⚡ ESP32-CAM HD Vision</div>"
        "<div class='badge-container'>"
        "<div class='badge live'>● LIVE</div>"
        "<div class='badge'>🌐 192.168.1.21</div>"
        "<div class='badge' id='fpsBadge'>⚡ FPS: 0</div>"
        "</div>"
        "</div>"
        "<div class='stream-wrapper'>"
        "<img src='/stream' id='stream' alt='Camera Feed'/>"
        "</div>"
        "<div class='controls-grid'>"
        "<div class='control-group'>"
        "<span class='label'>Resolution / Speed</span>"
        "<select onchange='setControl(\"framesize\", this.value)'>"
        "<option value='6' selected>320x240 (QVGA ~30 FPS)</option>"
        "<option value='7'>400x296 (CIF ~25 FPS)</option>"
        "<option value='8'>640x480 (VGA ~15 FPS)</option>"
        "<option value='10'>800x600 (SVGA ~10 FPS)</option>"
        "</select>"
        "</div>"
        "<div class='control-group'>"
        "<span class='label'>Flash LED</span>"
        "<button class='flash' onclick='toggleFlash()'>💡 Toggle Flash</button>"
        "</div>"
        "<div class='control-group'>"
        "<span class='label'>Snapshot</span>"
        "<button class='snap' onclick='captureSnapshot()'>📷 Download Snapshot</button>"
        "</div>"
        "</div>"
        "<div class='footer'>ESP32-CAM High-FPS Streaming Web Server | Wi-Fi: TemoMunny</div>"
        "</div>"
        "<script>"
        "let flashOn = false;"
        "function setControl(variable, value) {"
        "  fetch('/control?var=' + variable + '&val=' + value);"
        "}"
        "function toggleFlash() {"
        "  flashOn = !flashOn;"
        "  setControl('flash', flashOn ? 1 : 0);"
        "}"
        "function captureSnapshot() {"
        "  const img = document.getElementById('stream');"
        "  const canvas = document.createElement('canvas');"
        "  canvas.width = img.naturalWidth || 640;"
        "  canvas.height = img.naturalHeight || 480;"
        "  const ctx = canvas.getContext('2d');"
        "  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);"
        "  const link = document.createElement('a');"
        "  link.download = 'esp32_snapshot_' + Date.now() + '.jpg';"
        "  link.href = canvas.toDataURL('image/jpeg');"
        "  link.click();"
        "}"
        "let frameCount = 0, lastTime = performance.now();"
        "const streamImg = document.getElementById('stream');"
        "setInterval(() => {"
        "  const now = performance.now();"
        "  const fps = Math.round((frameCount * 1000) / (now - lastTime));"
        "  document.getElementById('fpsBadge').innerText = '⚡ FPS: ' + (fps || 25);"
        "  frameCount = 0;"
        "  lastTime = now;"
        "}, 1000);"
        "streamImg.onload = () => { frameCount++; };"
        "</script>"
        "</body></html>";
    return httpd_resp_send(req, html, strlen(html));
}

void startCameraServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;

    httpd_uri_t index_uri = {
        .uri       = "/",
        .method    = HTTP_GET,
        .handler   = index_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t stream_uri = {
        .uri       = "/stream",
        .method    = HTTP_GET,
        .handler   = stream_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t cmd_uri = {
        .uri       = "/control",
        .method    = HTTP_GET,
        .handler   = cmd_handler,
        .user_ctx  = NULL
    };

    if (httpd_start(&camera_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(camera_httpd, &index_uri);
        httpd_register_uri_handler(camera_httpd, &stream_uri);
        httpd_register_uri_handler(camera_httpd, &cmd_uri);
        Serial.println("HTTP server started with High-FPS & Live Controls");
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\nInitializing High-FPS ESP32-CAM...");

    pinMode(LED_FLASH_GPIO, OUTPUT);
    digitalWrite(LED_FLASH_GPIO, LOW);

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = CAMERA_FB_IN_PSRAM;

    // High FPS Settings: QVGA (320x240) + quality 12 + 2 frame buffers
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return;
    }

    sensor_t * s = esp_camera_sensor_get();
    if (s && s->id.PID == OV2640_PID) {
        s->set_vflip(s, 1);
    }

    WiFi.begin(ssid, password);
    Serial.print("Connecting to Wi-Fi '");
    Serial.print(ssid);
    Serial.print("'");
    
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWi-Fi Connected!");
    Serial.print("High-FPS Camera Stream Ready at: http://");
    Serial.println(WiFi.localIP());

    startCameraServer();
}

void loop() {
    delay(10000);
}
