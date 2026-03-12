#include <Arduino.h>
#include <SPI.h>
#include <AccelStepper.h>
#include "RotaryEncoder.h"

// --- Pins IHM01A1 (Arduino-Header / Nucleo-Arduino-Layout) ---
const int ENC_A_PIN = D4;
const int ENC_B_PIN = D5;

const int L6474_FLAG_PIN   = D2;
const int L6474_STBYRST    = D8;   // STBY\RST
const int L6474_DIR        = D7;   // DIR1
const int L6474_STEP       = D9;   // PWM1 = STEP-CLOCK
const int L6474_CS         = D10;  // SPI CS

static const unsigned int BAUD_RATE = 1000000;

// --- Stepper / Encoder constants ---
const int ENC_STEPS_PER_ROTATION = 1200;
const int STP_STEPS_PER_ROTATION = 200;
const unsigned int MICROSTEP_DIV = 16;
const int STEPS_PER_REV = STP_STEPS_PER_ROTATION * MICROSTEP_DIV; // 3200

// --- AccelStepper (DRIVER: STEP+DIR) ---
AccelStepper stp(AccelStepper::DRIVER, L6474_STEP, L6474_DIR);

// --- Encoder ---
RotaryEncoder *encoder = nullptr;

// ---------- L6474 low-level SPI helpers ----------
static inline void l6474_xfer(const uint8_t *tx, uint8_t *rx, size_t n) {
  digitalWrite(L6474_CS, LOW);
  for (size_t i = 0; i < n; i++) {
    uint8_t r = SPI.transfer(tx ? tx[i] : 0);
    if (rx) rx[i] = r;
  }
  digitalWrite(L6474_CS, HIGH);
}

// Commands (Enable/Disable): siehe “enable and disable commands” (Bitstruktur im Datasheet)
static const uint8_t L6474_CMD_DISABLE = 0xA8; // 1010 1000
static const uint8_t L6474_CMD_ENABLE  = 0xB8; // 1011 1000

static inline void l6474_disable() {
  uint8_t c = L6474_CMD_DISABLE;
  l6474_xfer(&c, nullptr, 1);
}

static inline void l6474_enable() {
  uint8_t c = L6474_CMD_ENABLE;
  l6474_xfer(&c, nullptr, 1);
}

// SetParam: 0b000xxxxx (addr in low 5 bits) :contentReference[oaicite:6]{index=6}
static inline void l6474_setParam(uint8_t addr, uint32_t value, uint8_t nbytes) {
  uint8_t buf[4];
  buf[0] = (uint8_t)(0x00 | (addr & 0x1F));
  // MSB first
  for (uint8_t i = 0; i < nbytes; i++) {
    buf[1 + i] = (uint8_t)(value >> (8 * (nbytes - 1 - i)));
  }
  l6474_xfer(buf, nullptr, 1 + nbytes);
}

// L6474 Register addresses (aus Register Map)
static const uint8_t L6474_REG_TVAL      = 0x09; // TVAL :contentReference[oaicite:7]{index=7}
static const uint8_t L6474_REG_STEP_MODE = 0x16; // STEP_MODE :contentReference[oaicite:8]{index=8}

static inline uint8_t tval_from_mA(uint16_t mA) {
  // TVAL LSB = 31.25 mA (7-bit)
  // clamp 31..4000 mA
  if (mA < 31) mA = 31;
  if (mA > 4000) mA = 4000;
  // round
  uint16_t code = (uint16_t)((mA + 15) / 31);
  if (code > 0x7F) code = 0x7F;
  return (uint8_t)code;
}

static inline float enc_deg() {
  int p = encoder->getPosition() % ENC_STEPS_PER_ROTATION;
  if (p < 0) p += ENC_STEPS_PER_ROTATION;
  return p * (360.0f / ENC_STEPS_PER_ROTATION);
}

static inline float stp_deg() {
  long p = stp.currentPosition() % STEPS_PER_REV;
  if (p < 0) p += STEPS_PER_REV;
  return p * (360.0f / (float)STEPS_PER_REV);
}

static inline float degps_to_steps_per_s(float deg_per_s) {
  // steps/s = deg/s * STEPS_PER_REV / 360
  return fabsf(deg_per_s) * (float)STEPS_PER_REV / 360.0f;
}

void encoderISR() { encoder->tick(); }

// --- Serial line reader ---
String readLine() {
  if (!Serial.available()) return String();
  String s = Serial.readStringUntil('\n');
  s.trim();
  return s;
}

// --- State ---
static bool loop_running = false;
static uint32_t t0_ms = 0;

static float stp_vel_degps = 0.0f;   // signed deg/s
static float enc_zero = 0.0f;   // signed deg/s

void setup() {
  Serial.begin(BAUD_RATE);
  Serial.setTimeout(20);

  pinMode(ENC_A_PIN, INPUT_PULLUP);
  pinMode(ENC_B_PIN, INPUT_PULLUP);

  encoder = new RotaryEncoder(ENC_A_PIN, ENC_B_PIN, RotaryEncoder::LatchMode::TWO03);
  attachInterrupt(digitalPinToInterrupt(ENC_A_PIN), encoderISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_B_PIN), encoderISR, CHANGE);

  pinMode(L6474_CS, OUTPUT);
  digitalWrite(L6474_CS, HIGH);

  pinMode(L6474_STBYRST, OUTPUT);
  digitalWrite(L6474_STBYRST, LOW);   // force standby/reset
  delay(5);
  digitalWrite(L6474_STBYRST, HIGH);  // exit standby
  // Datasheet: nach Exit tlogicwu + tcpwu warten :contentReference[oaicite:9]{index=9}
  delay(10);

  SPI.begin();
  SPI.beginTransaction(SPISettings(5000000, MSBFIRST, SPI_MODE3)); // L6474 SPI (Mode 3 typisch)

  // 1) Sicher in HiZ
  l6474_disable();

  // 2) Parameter setzen (nur sinnvoll/erlaubt in HiZ für STEP_MODE) :contentReference[oaicite:10]{index=10}
  // STEP_MODE: Bit7 muss beim Schreiben 1 sein; STEP_SEL = 1/16 => 1xx
  // simplest: 0b1000 0111 = 0x87 (setzt Bit7=1, STEP_SEL auf 1/16, SYNC_SEL=0)
  l6474_setParam(L6474_REG_STEP_MODE, 0x87, 1);

  // TVAL: z.B. 500 mA -> Code
  uint8_t tval = tval_from_mA(500);
  l6474_setParam(L6474_REG_TVAL, tval, 1);

  // 3) Bridges aktivieren
  l6474_enable();

  // AccelStepper settings
  pinMode(L6474_STEP, OUTPUT);
  pinMode(L6474_DIR, OUTPUT);

  stp.setMaxSpeed(2000);      // steps/s (microsteps/s)
  stp.setSpeed(0);

  loop_running = false;
}
static uint32_t next_tick_ms = 0;
static const uint32_t dt_ms = 10;

void loop() {
  uint32_t now = millis();

  // Stepper muss permanent gepumpt werden
  if (loop_running) {
    stp.runSpeed();
  }

  // --- Kommando lesen (optional, blockiert nur kurz wegen Timeout) ---
  String line = readLine();   // liefert "" wenn nichts da

  // --- State machine / commands ---
  if (!loop_running) {
    if (line.length()) {
      if (line == "START") {
        t0_ms = now;
        next_tick_ms = now + dt_ms;
        loop_running = true;

        // sicherheitshalber: Bridges aktiv (falls vorher STOP disabled)
        l6474_enable();

        stp_vel_degps = 1.0f;
        float sps = degps_to_steps_per_s(stp_vel_degps);
        stp.setMaxSpeed(fabs(sps));
        stp.setSpeed((stp_vel_degps >= 0) ? sps : -sps);

        Serial.println("START");
      }
      else if (line == "HOME") {
        stp.setCurrentPosition(180 * (float)STEPS_PER_REV / 360.0f);
        Serial.println("HOME SET");
      }
      else if (line.startsWith("CAL ")) {
        enc_zero = line.substring(4).toFloat();
        noInterrupts();
        encoder->setPosition(enc_zero * (float)ENC_STEPS_PER_ROTATION / 360.0f);
        interrupts();
        Serial.println("ENC ZERO");
      }
    }
    return;
  }

  // running: commands
  if (line.length()) {
    if (line.startsWith("VEL ")) {
      stp_vel_degps = line.substring(4).toFloat();
      float sps = degps_to_steps_per_s(stp_vel_degps);

      stp.setMaxSpeed(fabs(sps));
      stp.setSpeed((stp_vel_degps >= 0) ? sps : -sps);

      Serial.println("VELOCITY SET");
    }
    else if (line == "STOP") {
      stp.setSpeed(0);
      loop_running = false;

      // optional: Bridges in HiZ
      l6474_disable();

      Serial.println("STOP");
      return;
    }
  }

  // --- Telemetrie alle 20ms ---
  if (now >= next_tick_ms) {
    uint32_t t_ms = now - t0_ms;
    Serial.print(t_ms);
    Serial.write('\t'); Serial.print(enc_deg());
    Serial.write('\t'); Serial.print(stp_deg());
    Serial.write('\t'); Serial.println(stp_vel_degps, 4);

    next_tick_ms += dt_ms;
  }
}

