#include "nvs_flash.h"
#include "protocol_examples_common.h"
#include "esp_log.h"
#include "esp_mqtt.hpp"
#include "esp_mqtt_client_config.hpp"
#include "freertos/task.h"
#include <cmath>
#include <chrono>
#include <cJSON.h>
#include <esp_wifi_types_generic.h>
#include <esp_wifi.h>

namespace mqtt = idf::mqtt;

namespace {
    constexpr auto *TAG = "DATA_COLLECTOR";
    constexpr float PUBLISH_FREQUENCY = 0.01;
    constexpr float SINE_FREQUENCY = M_2_PI / 5;
    constexpr float PHASE_SHIFT = M_PI_2;

    class MyClient final : public mqtt::Client {
    public:
        using mqtt::Client::Client;

    private:
        void on_connected(esp_mqtt_event_handle_t const event) override {
            xTaskCreate(
                    [](void *param) {
                        auto *client = static_cast<MyClient *>(param);
                        TickType_t last_wake_time = xTaskGetTickCount();
                        constexpr TickType_t frequency = pdMS_TO_TICKS(
                                1000 * PUBLISH_FREQUENCY);

                        while (true) {
                            client->publish_data();
                            vTaskDelayUntil(&last_wake_time, frequency);  // Precise delay for periodic execution
                        }
                    },
                    "DataPublisherTask", 4096, this, 5, nullptr
            );
        }

        void on_data(esp_mqtt_event_handle_t const event) override {}

        void publish_data() {
            using namespace std::chrono;
            char timestamp[30];

            // Get current time and format it
            auto now = std::chrono::system_clock::now();
            auto now_time_t = std::chrono::system_clock::to_time_t(now);
            auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1000000;

            std::tm local_time;
            localtime_r(&now_time_t, &local_time);

            // Format time with seconds and microseconds precision
            strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &local_time);
            snprintf(timestamp + 19, sizeof(timestamp) - 19, ".%06lld", now_us.count());

            // Calculate elapsed time in seconds
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
            float time_in_seconds = elapsed_ms / 1000.0f;

            // Generate sine values based on time for a 5-second period
            float x = std::sin(SINE_FREQUENCY * time_in_seconds);
            float y = std::sin(SINE_FREQUENCY * time_in_seconds + PHASE_SHIFT);
            float z = std::sin(SINE_FREQUENCY * time_in_seconds + 2 * PHASE_SHIFT);

            // Create JSON object
            cJSON *json = cJSON_CreateObject();
            cJSON_AddStringToObject(json, "timestamp", timestamp);
            cJSON_AddNumberToObject(json, "x", x);
            cJSON_AddNumberToObject(json, "y", y);
            cJSON_AddNumberToObject(json, "z", z);

            // Convert JSON object to string
            char *payload = cJSON_PrintUnformatted(json);
            auto message = mqtt::StringMessage{
                .data = payload,
                .qos = mqtt::QoS::AtMostOnce
            };

            // Publish the JSON payload
            publish("rattechan/sensor", message);

            // Free the JSON object and string
            cJSON_Delete(json);
            free(payload);
        }
    };
}

extern "C" [[noreturn]] void app_main(void) {
    ESP_LOGI(TAG, "[APP] Startup..");
    ESP_LOGI(TAG, "[APP] Free memory: %" PRIu32 " bytes", esp_get_free_heap_size());
    ESP_LOGI(TAG, "[APP] IDF version: %s", esp_get_idf_version());

    esp_log_level_set("*", ESP_LOG_INFO);

    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(example_connect());
    esp_wifi_set_ps(WIFI_PS_NONE);

    mqtt::BrokerConfiguration broker{
            .address = {mqtt::URI{std::string{CONFIG_BROKER_URL}}},
            .security = mqtt::Insecure{}
    };
    mqtt::ClientCredentials credentials{};
    mqtt::Configuration config{};
    MyClient client{broker, credentials, config};

    while (true) {
        constexpr TickType_t xDelay = 500 / portTICK_PERIOD_MS;
        vTaskDelay(xDelay);
    }
}
