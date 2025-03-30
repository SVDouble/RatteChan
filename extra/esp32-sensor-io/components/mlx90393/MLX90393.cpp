#include "MLX90393.hpp"
#include "driver/i2c_master.h"
#include "esp_log.h"

// Constructor for I2C communication
MLX90393::MLX90393(i2c_port_t i2c_port, uint8_t i2c_address) : i2c_port(i2c_port), i2c_address(i2c_address) {
    i2c_config_t config = {
            .mode = I2C_MODE_MASTER,
            .sda_io_num = CONFIG_PIN_NUM_SDA,
            .scl_io_num = CONFIG_PIN_NUM_SCL,
            .sda_pullup_en = GPIO_PULLUP_ENABLE,
            .scl_pullup_en = GPIO_PULLUP_ENABLE,
            .master = {
                    .clk_speed = 400000
            }
    };
    i2c_param_config(i2c_port, &config);
    i2c_driver_install(i2c_port, config.mode, 0, 0, 0);
}

// *************************************** MAIN FUNCTIONS ***************************************

void MLX90393::EX(char *receiveBuffer) {
    write_buffer[0] = 0x80;
    Send_I2C(receiveBuffer, write_buffer, 1, 1);
}

void MLX90393::SB(char *receiveBuffer, char zyxt) {
    write_buffer[0] = (0x10) | (zyxt);
    Send_I2C(receiveBuffer, write_buffer, 1, 1);
}

void MLX90393::SWOC(char *receiveBuffer, char zyxt) {
    write_buffer[0] = (0x20) | (zyxt);
    Send_I2C(receiveBuffer, write_buffer, 1, 1);
}

void MLX90393::SM(char *receiveBuffer, char zyxt) {
    write_buffer[0] = (0x30) | (zyxt);
    Send_I2C(receiveBuffer, write_buffer, 1, 1);
}

void MLX90393::RM(char *receiveBuffer, char zyxt) {
    write_buffer[0] = (0x40) | (zyxt);
    for (int i = 0; i < 2 * count_set_bits(zyxt); i++) {
        write_buffer[i + 2] = 0x00;
    }
    Send_I2C(receiveBuffer, write_buffer, 1, 1 + 2 * count_set_bits(zyxt));
}

void MLX90393::RR(char *receiveBuffer, int address) {
    write_buffer[0] = 0x50;
    write_buffer[1] = address << 2;
    Send_I2C(receiveBuffer, write_buffer, 2, 3);
}

void MLX90393::WR(char *receiveBuffer, int address, int data) {
    write_buffer[0] = 0x60;
    write_buffer[1] = (data & 0xFF00) >> 8;
    write_buffer[2] = data & 0x00FF;
    write_buffer[3] = address << 2;
    Send_I2C(receiveBuffer, write_buffer, 4, 1);
}

void MLX90393::RT(char *receiveBuffer) {
    write_buffer[0] = 0xF0;
    Send_I2C(receiveBuffer, write_buffer, 1, 1);
}

void MLX90393::NOP(char *receiveBuffer) {
    write_buffer[0] = 0x00;
    Send_I2C(receiveBuffer, write_buffer, 1, 1);
}

void MLX90393::HR(char *receiveBuffer) {
    write_buffer[0] = 0xD0;
    Send_I2C(receiveBuffer, write_buffer, 1, 1);
}

void MLX90393::HS(char *receiveBuffer) {
    write_buffer[0] = 0xE0;
    Send_I2C(receiveBuffer, write_buffer, 1, 1);
}

// ************************************* COMMUNICATION LEVEL ************************************

void MLX90393::Send_I2C(char *receiveBuffer, char *sendBuffer, int sendMessageLength, int receiveMessageLength) {
    esp_err_t write_err = i2c_master_write_to_device(i2c_port, i2c_address, (uint8_t *) sendBuffer, sendMessageLength,
                                                     pdMS_TO_TICKS(1000));
    if (write_err != ESP_OK) {
        ESP_LOGW("MLX90393", "I2C write failed with error: %s", esp_err_to_name(write_err));
    }

    esp_err_t read_err = i2c_master_read_from_device(i2c_port, i2c_address, (uint8_t *) receiveBuffer,
                                                     receiveMessageLength,
                                                     pdMS_TO_TICKS(1000));
    if (read_err != ESP_OK) {
        ESP_LOGW("MLX90393", "I2C read failed with error: %s", esp_err_to_name(read_err));
    }
}

// *************************************** EXTRA FUNCTIONS **************************************

int MLX90393::count_set_bits(int N) {
    int result = 0;
    while (N) {
        result++;
        N &= N - 1;
    }
    return result;
}
