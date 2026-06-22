#include <Servo.h>

Servo servoTilt;   // pin9
Servo servoPan;    // pin10

int tiltPin = 9;
int panPin = 10;

const int scanStart = 50;
const int scanEnd = 180;
const int scanStep = 2;

// 中央位置（実機に合わせて調整）
const int centerPos1 = 90;   // Tilt中央
const int centerPos2 = 90;   // Pan中央

const int settleTime = 50;
const int photoDelay = 500;


void setup() {
  Serial.begin(9600);
  
  // attachはsetup()の中、パルス幅を指定
  servoPan.attach(panPin, 1000, 2000);
  servoTilt.attach(tiltPin, 1000, 2000);

  // 初期位置
  servoPan.write(centerPos2);
  servoTilt.write(centerPos1);

  delay(2000);
}

void loop() {
  Serial.println("--- GLOBAL_SCAN_START ---");

  // =====================
  //  フェーズ1：Tilt(pin9)のみ動かす
  //  Pan(pin10)は中央固定
  // =====================
  Serial.println("--- PHASE1: TILT SCAN ---");

  for (int tilt = scanStart; tilt <= scanEnd; tilt += scanStep) {

    servoPan.write(centerPos2);   // 常に中央維持
    servoTilt.write(tilt);
    delay(settleTime);

    Serial.print("SHOT:T");
    Serial.print(tilt);
    Serial.print(":P");
    Serial.println(centerPos2);

    delay(photoDelay);

    while (Serial.available() > 0) {
      Serial.read();
    }
  }

  // Tiltを中央へ戻す
  servoTilt.write(centerPos1);
  delay(1500);

  // =====================
  //  フェーズ1：Pan(pin10)のみ動かす
  //  Tilt(pin9)は中央固定
  // =====================
  Serial.println("--- PHASE2: PAN SCAN ---");

  for (int pan = scanStart; pan <= scanEnd; pan += scanStep) {

    servoTilt.write(centerPos1);   // 常に中央維持
    servoPan.write(pan);
    delay(settleTime);

    Serial.print("SHOT:T");
    Serial.print(centerPos1);
    Serial.print(":P");
    Serial.println(pan);

    delay(photoDelay);

    while (Serial.available() > 0) {
      Serial.read();
    }
  }

  // Panを中央へ戻す
  servoPan.write(centerPos2);
  delay(1500);

  Serial.println("--- GLOBAL_SCAN_END ---");

  // 初期位置へ戻す
  servoTilt.write(centerPos1);
  servoPan.write(centerPos2);
  delay(1000);
}