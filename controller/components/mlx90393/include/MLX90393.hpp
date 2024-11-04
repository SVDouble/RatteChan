#ifndef ESP_IDF_MLX90393_H
#define ESP_IDF_MLX90393_H

#include "driver/i2c.h"

/** MLX90393 class for ESP-IDF.
 *  Used for communication with the MLX90393 in I2C
 */
class MLX90393 {
        public:

        /** [Constructor] Create MLX90393 instance, using I2C for communication.
         * @param i2c_port I2C port number (I2C_NUM_0 or I2C_NUM_1).
         * @param i2c_address I2C address of the MLX90393.
         */
        MLX90393(i2c_port_t i2c_port, uint8_t i2c_address);

        /** Send 'exit' command to MLX90393.
         * @param *receiveBuffer Pointer to receiveBuffer, will contain response of IC after command is sent.
         * @note The receiveBuffer will contain the status byte only.
         */
        void EX(char *receiveBuffer);

        /** Send 'start burst mode' command to MLX90393.
         * @param *receiveBuffer Pointer to receiveBuffer, will contain response of IC after command is sent.
         * @param zyxt Selection of the axes/temperature to be measured.
         * @note The receiveBuffer will contain the status byte only.
         */
        void SB(char *receiveBuffer, char zyxt);

        /** Send 'start wake-up on change mode' command to MLX90393.
         * @param *receiveBuffer Pointer to receiveBuffer, will contain response of IC after command is sent.
         * @param zyxt Selection of the axes/temperature to which the mode should apply.
         * @note The receiveBuffer will contain the status byte only.
         */
        void SWOC(char *receiveBuffer, char zyxt);

        /** Send 'single measurement' command to MLX90393.
         * @param *receiveBuffer Pointer to receiveBuffer, will contain response of IC after command is sent.
         * @param zyxt Selection of the axes/temperature to be measured.
         * @note The receiveBuffer will contain the status byte only.
         */
        void SM(char *receiveBuffer, char zyxt);

        /** Send 'read measurement' command to MLX90393.
         * @param *receiveBuffer Pointer to receiveBuffer, will contain response of IC after command is sent.
         * @param zyxt Selection of the axes/temperature to be read out.
         * @note The receiveBuffer will contain the status byte, followed by 2 bytes for each T, X, Y, and Z.
         */
        void RM(char *receiveBuffer, char zyxt);

        /** Send 'read register' command to MLX90393.
         * @param *receiveBuffer Pointer to receiveBuffer, will contain response of IC after command is sent.
         * @param address The register to be read out.
         * @note The receiveBuffer will contain the status byte, followed by 2 bytes for the data at the specific register.
         */
        void RR(char *receiveBuffer, int address);

        /** Send 'write register' command to MLX90393.
         * @param *receiveBuffer Pointer to receiveBuffer, will contain response of IC after command is sent.
         * @param address The register to be written.
         * @param data The 16-bit word to be written in the register.
         * @note The receiveBuffer will only contain the status byte.
         */
        void WR(char *receiveBuffer, int address, int data);

        /** Send 'reset' command to MLX90393.
         * @param *receiveBuffer Pointer to receiveBuffer, will contain response of IC after command is sent.
         * @note The receiveBuffer will contain the status byte only.
         */
        void RT(char *receiveBuffer);

        /** Send 'NOP' command to MLX90393.
         * @param *receiveBuffer Pointer to receiveBuffer, will contain response of IC after command is sent.
         * @note The receiveBuffer will contain the status byte only.
         */
        void NOP(char *receiveBuffer);

        /** Send 'memory recall' command to MLX90393.
         * @param *receiveBuffer Pointer to receiveBuffer, will contain response of IC after command is sent.
         * @note The receiveBuffer will contain the status byte only.
         */
        void HR(char *receiveBuffer);

        /** Send 'memory store' command to MLX90393.
         * @param *receiveBuffer Pointer to receiveBuffer, will contain response of IC after command is sent.
         * @note The receiveBuffer will contain the status byte only.
         */
        void HS(char *receiveBuffer);

        int count_set_bits(int N);

        private:
        // I2C
        i2c_port_t i2c_port;
        uint8_t i2c_address;
        void Send_I2C(char *receiveBuffer, char *sendBuffer, int sendMessageLength, int receiveMessageLength);

        // Shared
        char write_buffer[10];
};

#endif
