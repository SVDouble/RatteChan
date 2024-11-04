#include "nvs_flash.h"
#include "protocol_examples_common.h"
#include "esp_log.h"
#include "esp_mqtt.hpp"
#include "esp_mqtt_client_config.hpp"
#include "freertos/task.h"
#include <cmath>
#include <cJSON.h>
#include <esp_wifi_types_generic.h>
#include <esp_wifi.h>
#include <esp_timer.h>
#include "freertos/semphr.h"

namespace mqtt = idf::mqtt;

namespace {
    constexpr auto *TAG = "DATA_COLLECTOR";
    constexpr float PUBLISH_FREQUENCY = 0.01;
    constexpr float SINE_FREQUENCY = M_2_PI * 1.1;
    constexpr float PHASE_SHIFT = M_PI_2;

    class MyClient final : public mqtt::Client {
    public:
        using mqtt::Client::Client;

    private:
        // Success counter and mutex for thread-safe access
        int publish_success_count = 0;
        SemaphoreHandle_t publish_mutex = xSemaphoreCreateMutex();

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

            // Start the statistics logging task
            xTaskCreate(
                    [](void *param) {
                        auto *client = static_cast<MyClient *>(param);

                        while (true) {
                            vTaskDelay(pdMS_TO_TICKS(1000));  // 1-second interval
                            client->log_publish_stats();
                        }
                    },
                    "PublishStatsTask", 2048, this, 5, nullptr
            );
        }

        void on_data(esp_mqtt_event_handle_t const event) override {}

        void publish_data() {
            // Get seconds since start
            float t = esp_timer_get_time() / 1000000.0f;  // Convert microseconds to seconds

            // Generate sine values based on seconds_since_start for a 5-second period
            float x = std::sin(SINE_FREQUENCY * t);
            float y = std::sin(SINE_FREQUENCY * t + PHASE_SHIFT);
            float z = std::sin(SINE_FREQUENCY * t + 2 * PHASE_SHIFT);

            // Create JSON object
            cJSON *json = cJSON_CreateObject();
            cJSON_AddNumberToObject(json, "t", t);
            cJSON_AddNumberToObject(json, "x", x);
            cJSON_AddNumberToObject(json, "y", y);
            cJSON_AddNumberToObject(json, "z", z);

            // Convert JSON object to string
            char *payload = cJSON_PrintUnformatted(json);
            auto message = mqtt::StringMessage{
                    .data = payload,
                    .qos = mqtt::QoS::AtMostOnce
            };

            // Publish the JSON payload and check if it succeeds
            if (publish("ratte/data", message) != std::nullopt) {
                // Thread-safe increment of publish success count
                xSemaphoreTake(publish_mutex, portMAX_DELAY);
                publish_success_count++;
                xSemaphoreGive(publish_mutex);
            } else {
                ESP_LOGW(TAG, "Failed to publish message to topic: ratte/data");
            }

            // Free the JSON object and string
            cJSON_Delete(json);
            free(payload);
        }

        void log_publish_stats() {
            // Thread-safe access to publish_success_count
            xSemaphoreTake(publish_mutex, portMAX_DELAY);
            int success_count = publish_success_count;
            publish_success_count = 0;  // Reset counter
            xSemaphoreGive(publish_mutex);

            // Log the success count
            ESP_LOGI(TAG, "Publishes per second: %d", success_count);
        }

    };
}

extern "C" [[noreturn]] void app_main(void) {
    ESP_LOGI(TAG, "[APP] Startup..");
    ESP_LOGI(TAG, "[APP] Free memory: %" PRIu32 " bytes", esp_get_free_heap_size());
    ESP_LOGI(TAG, "[APP] IDF version: %s", esp_get_idf_version());

    esp_log_level_set("*", ESP_LOG_INFO);
    esp_log_level_set("MQTT_CLIENT", ESP_LOG_VERBOSE);

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
