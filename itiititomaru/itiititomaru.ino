#include <Servo.h>

Servo servoPan;
Servo servoTilt;

int panPin = 10;
int tiltPin = 9;

const int scanStart = 70;
const int scanEnd = 200;
const int scanStep = 2;
const int centerPos1 = 115;  // Tiltの真ん中
const int centerPos2 = 145;  // Panの真ん中
const int settleTime = 50;
const int photoDelay = 500;

void setup() {
  Serial.begin(9600);
  
  // attachはsetup()の中、パルス幅を指定
  servoPan.attach(panPin, 500, 2500);
  servoTilt.attach(tiltPin, 500, 2500);

  servoPan.write(centerPos2);
  servoTilt.write(scanStart);
  delay(2000);
}

void loop() {
  Serial.println("--- GLOBAL_SCAN_START ---");

  // =====================
  // フェーズ1：Tiltを70→200までスキャン（Panは175固定）
  // =====================
  Serial.println("--- PHASE1: TILT SCAN ---");
  
  for (int tilt = scanStart; tilt <= scanEnd; tilt += scanStep) {
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

  servoTilt.write(centerPos1);
  delay(1000);

  // =====================
  // フェーズ2：Panを70→200までスキャン（Tiltは150固定）
  // =====================
  Serial.println("--- PHASE2: PAN SCAN ---");

  for (int pan = scanStart; pan <= scanEnd; pan += scanStep) {
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

  servoPan.write(centerPos2);
  delay(1000);

  Serial.println("--- GLOBAL_SCAN_END ---");

  servoTilt.write(scanStart);
  servoPan.write(centerPos2);
  delay(10000);
}