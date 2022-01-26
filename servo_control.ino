#include <Servo.h>

Servo myservo;

#define servoPin 9


int value = 0;
int angle = 0;


void setup() {
  Serial.begin(9600);
  myservo.attach(servoPin);
  myservo.write(90);
  delay(1000);
}


void loop()
{
  if (Serial.available() > 0)
  {
    int value = char(Serial.read()) - '0';
    if (value == 1)
    {
      for (angle = 90; angle <= 150; angle += 1) {
      myservo.write(angle);
      delay(7);
      }
      for (angle = 150; angle >= 90; angle -= 1) {
      myservo.write(angle);
      delay(7);
      }
      //exit(0);
    }
    else if (value == 0)
    {
      for (angle = 90; angle >= 30; angle -= 1) {
      myservo.write(angle);
      delay(7);
      }
      for (angle = 30; angle <= 90; angle += 1) {
      myservo.write(angle);
      delay(7);
      }
      //exit(0);
    }
  }
}
