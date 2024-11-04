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

#include "MLX90393.hpp"

namespace mqtt = idf::mqtt;

namespace {
    constexpr auto *TAG = "DATA_COLLECTOR";
    constexpr float PUBLISH_FREQUENCY = 0.01;
    constexpr uint8_t I2C_ADDRESS = 0x18; // Address with A0 and A1 grounded

    // Create an MLX90393 sensor instance on I2C port 0
    MLX90393 sensor(I2C_NUM_0, I2C_ADDRESS);

    class MyClient final : public mqtt::Client {
    public:
        using mqtt::Client::Client;

    private:
        int publish_success_count = 0;
        SemaphoreHandle_t publish_mutex = xSemaphoreCreateMutex();

        void on_connected(esp_mqtt_event_handle_t const event) override {
            // Set up burst mode for continuous measurement on XYZ axes and temperature
            char burst_init_response[1];
            sensor.SB(burst_init_response, 0x0F);  // Start burst mode with XYZ and T

            // Check if BURST_MODE bit is set (bit 7)
            if (burst_init_response[0] & 0x80) {
                ESP_LOGI("MLX90393", "Burst mode successfully activated.");
            } else {
                ESP_LOGW("MLX90393", "Burst mode not activated. Check configuration.");
            }

            xTaskCreate(
                    [](void *param) {
                        auto *client = static_cast<MyClient *>(param);
                        TickType_t last_wake_time = xTaskGetTickCount();
                        constexpr TickType_t frequency = pdMS_TO_TICKS(1000 * PUBLISH_FREQUENCY);

                        // Start continuous burst mode for x, y, z, and temperature
                        char receiveBuffer[1];
                        sensor.SB(receiveBuffer, 0x0F);  // 0x0F enables XYZ and T in burst mode

                        while (true) {
                            client->publish_data();
                            vTaskDelayUntil(&last_wake_time, frequency);
                        }
                    },
                    "DataPublisherTask", 4096, this, 5, nullptr
            );

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
            float ts = esp_timer_get_time() / 1000000.0f;  // Convert microseconds to seconds
            char receiveBuffer[10];

            // Trigger a read of XYZ and Temperature data
            sensor.RM(receiveBuffer, 0x0F);

            // Extract sensor data
            int16_t x = (receiveBuffer[1] << 8) | receiveBuffer[2];
            int16_t y = (receiveBuffer[3] << 8) | receiveBuffer[4];
            int16_t z = (receiveBuffer[5] << 8) | receiveBuffer[6];
            int16_t t = (receiveBuffer[7] << 8) | receiveBuffer[8];

            // Create JSON payload
            cJSON *json = cJSON_CreateObject();
            cJSON_AddNumberToObject(json, "ts", ts);
            cJSON_AddNumberToObject(json, "x", x);
            cJSON_AddNumberToObject(json, "y", y);
            cJSON_AddNumberToObject(json, "z", z);
            cJSON_AddNumberToObject(json, "t", t);

            // Convert JSON object to string
            char *payload = cJSON_PrintUnformatted(json);
            auto message = mqtt::StringMessage{
                    .data = payload,
                    .qos = mqtt::QoS::AtMostOnce
            };

            if (publish("ratte/data", message) != std::nullopt) {
                xSemaphoreTake(publish_mutex, portMAX_DELAY);
                publish_success_count++;
                xSemaphoreGive(publish_mutex);
            } else {
                ESP_LOGW(TAG, "Failed to publish message to topic: ratte/data");
            }

            // Clean up JSON and buffer
            cJSON_Delete(json);
            free(payload);
        }

        void log_publish_stats() {
            xSemaphoreTake(publish_mutex, portMAX_DELAY);
            int success_count = publish_success_count;
            publish_success_count = 0;
            xSemaphoreGive(publish_mutex);

            ESP_LOGI(TAG, "Publishes per second: %d", success_count);
        }
    };
}

extern "C" [[noreturn]] void app_main(void) {
    ESP_LOGI(TAG, "[APP] Startup..");
    ESP_LOGI(TAG, "[APP] Free memory: %" PRIu32 " bytes", esp_get_free_heap_size());
    ESP_LOGI(TAG, "[APP] IDF version: %s", esp_get_idf_version());

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
