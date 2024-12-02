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
#include <lwip/sockets.h>

#include "freertos/semphr.h"

#include "MLX90393.hpp"
#include "../../../../../../../opt/esp-idf/components/mqtt/esp-mqtt/lib/include/mqtt_client_priv.h"
#include "../../../../../../../opt/esp-idf/components/tcp_transport/private_include/esp_transport_internal.h"

namespace mqtt = idf::mqtt;

namespace {
    constexpr auto *TAG = "RATTE";
    constexpr float PUBLISH_FREQUENCY_HZ = 100;
    constexpr uint8_t I2C_ADDRESS_0 = 0x19; // 0x18 - Address with A0 and A1 grounded ('0b11000'), 0x19 - A0=1
    constexpr uint8_t I2C_ADDRESS_1 = 0x18;

    // Create an MLX90393 sensor instance on I2C port 0
    MLX90393 sensor0(I2C_NUM_0, I2C_ADDRESS_0);
    MLX90393 sensor1(I2C_NUM_0, I2C_ADDRESS_1);

    class MyClient final : public mqtt::Client {
    public:
        using mqtt::Client::Client;

    private:
        int publish_success_count = 0;
        SemaphoreHandle_t publish_mutex = xSemaphoreCreateMutex();

        void on_connected(esp_mqtt_event_handle_t const event) override {
            // Disable Nagle's algorithm
            const auto t = handler->transport;
            const int socket = t->_get_socket(t);
            constexpr int flag = 1;
            if (const int r = setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(int)); r < 0) {
                ESP_LOGE(TAG, "setsockopt failed: %s", esp_err_to_name(r));
            }

            // Set up burst mode for continuous measurement on XYZ axes and temperature
            char sensor0_buffer[1], sensor1_buffer[1];
            // Start burst mode with XYZ and T
            sensor0.SB(sensor0_buffer, 0x0F);
            sensor1.SB(sensor1_buffer, 0x0F);

            // Check if BURST_MODE bit is set (bit 7)
            if ((sensor0_buffer[0] & 0x80) && (sensor1_buffer[0] & 0x80)) {
                ESP_LOGI("MLX90393", "Burst mode successfully activated.");
            } else {
                ESP_LOGW("MLX90393", "Burst mode not activated. Check configuration.");
            }

            xTaskCreatePinnedToCore(
                    [](void *param) {
                        auto *client = static_cast<MyClient *>(param);
                        TickType_t last_wake_time = xTaskGetTickCount();
                        constexpr TickType_t frequency = pdMS_TO_TICKS(1000 / PUBLISH_FREQUENCY_HZ);

                        while (true) {
                            client->publish_data();
                            vTaskDelayUntil(&last_wake_time, frequency);
                        }
                    },
                    "DataPublisherTask", 4096, this, tskIDLE_PRIORITY + 1, nullptr, 1
            );

            xTaskCreate(
                    [](void *param) {
                        auto *client = static_cast<MyClient *>(param);
                        while (true) {
                            vTaskDelay(pdMS_TO_TICKS(1000));  // 1-second interval
                            client->log_publish_stats();
                        }
                    },
                    "PublishStatsTask", 2048, this, tskIDLE_PRIORITY + 1, nullptr
            );
        }

        void on_data(esp_mqtt_event_handle_t const event) override {}

        static void read_data(MLX90393 *sensor, cJSON *json, char const *name) {
            char buffer[10];
            sensor->RM(buffer, 0x0F); // TODO: remove t

            // Extract sensor data
            int16_t x = (buffer[1] << 8) | buffer[2];
            int16_t y = (buffer[3] << 8) | buffer[4];
            int16_t z = (buffer[5] << 8) | buffer[6];

            // Append sensor data to JSON with unique keys
            char buf[256];
            snprintf(buf, sizeof(buf), "%s_x", name);
            cJSON_AddNumberToObject(json, buf, x);

            snprintf(buf, sizeof(buf), "%s_y", name);
            cJSON_AddNumberToObject(json, buf, y);

            snprintf(buf, sizeof(buf), "%s_z", name);
            cJSON_AddNumberToObject(json, buf, z);
        }

        void publish_data() {
            cJSON *json = cJSON_CreateObject();

            float ts = esp_timer_get_time() / 1000000.0f;  // Convert microseconds to seconds
            cJSON_AddNumberToObject(json, "ts", ts);

            read_data(&sensor0, json, "s0");
            read_data(&sensor1, json, "s1");

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
    //esp_log_level_set("transport_base", ESP_LOG_VERBOSE);
    //esp_log_level_set("mqtt_client", ESP_LOG_VERBOSE);
    esp_log_level_set("random", ESP_LOG_INFO);

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
