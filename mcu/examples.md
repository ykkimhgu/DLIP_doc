# Examples

## GPIO

### Blinking LED

{% tabs %}
{% tab title="mbed" %}
```cpp
#include "mbed.h"

DigitalOut led(LED1);

int main() {
    while(1) {
        led = 1;
        wait(0.5)
        led=0;
        wait(0.5);
    }
}
```
{% endtab %}

{% tab title="EC" %}

{% endtab %}

{% tab title="Arduino" %}
```cpp
// constants won't change. Used here to set a pin number:
const int ledPin =  LED_BUILTIN;// the number of the LED pin

// Variables will change:
int ledState = LOW;             // ledState used to set the LED

// Generally, you should use "unsigned long" for variables that hold time
// The value will quickly become too large for an int to store
unsigned long previousMillis = 0;        // will store last time LED was updated

// constants won't change:
const long interval = 1000;           // interval at which to blink (milliseconds)

void setup() {
  // set the digital pin as output:
  pinMode(ledPin, OUTPUT);
}

void loop() {
  // here is where you'd put code that needs to be running all the time.

  // check to see if it's time to blink the LED; that is, if the difference
  // between the current time and last time you blinked the LED is bigger than
  // the interval at which you want to blink the LED.
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    // save the last time you blinked the LED
    previousMillis = currentMillis;

    // if the LED is off turn it on and vice-versa:
    if (ledState == LOW) {
      ledState = HIGH;
    } else {
      ledState = LOW;
    }

    // set the LED with the ledState of the variable:
    digitalWrite(ledPin, ledState);
  }
}
```
{% endtab %}
{% endtabs %}

### 

### LED with button

{% tabs %}
{% tab title="mbed" %}
```cpp
#include "mbed.h"

DigitalIn  button(USER_BUTTON);
DigitalOut led(LED1);

int main() {
    while(1) {
        if(!button) led = 1;
        else led = 0;
    }
}
```
{% endtab %}

{% tab title="EC" %}

{% endtab %}

{% tab title="Arduino" %}
```cpp

// constants won't change. They're used here to set pin numbers:
const int buttonPin = 2;     // the number of the pushbutton pin
const int ledPin =  13;      // the number of the LED pin

// variables will change:
int buttonState = 0;         // variable for reading the pushbutton status

void setup() {
  // initialize the LED pin as an output:
  pinMode(ledPin, OUTPUT);
  // initialize the pushbutton pin as an input:
  pinMode(buttonPin, INPUT);
}

void loop() {
  // read the state of the pushbutton value:
  buttonState = digitalRead(buttonPin);

  // check if the pushbutton is pressed. If it is, the buttonState is HIGH:
  if (buttonState == HIGH) {
    // turn LED on:
    digitalWrite(ledPin, HIGH);
  } else {
    // turn LED off:
    digitalWrite(ledPin, LOW);
  }
}
```
{% endtab %}
{% endtabs %}

### 

## Interrupt

### Button Interrupt

{% tabs %}
{% tab title="mbed" %}
```cpp
#include "mbed.h"

InterruptIn button(USER_BUTTON); 
DigitalOut  led(LED1);

void pressed()
{
    led = 1; 
}

void released(){
    led = 0;
}

int main()
{
    button.fall(&pressed);
    button.rise(&released);
    while (1);
}

```
{% endtab %}

{% tab title="EC" %}

{% endtab %}

{% tab title="Arduino" %}

{% endtab %}
{% endtabs %}

## PWM

{% tabs %}
{% tab title="mbed" %}
```cpp
#include "mbed.h"

Serial      pc(USBTX, USBRX, 9600);
PwmOut      trig(D10); // Trigger 핀
InterruptIn echo(D7);  // Echo 핀
Timer       tim;

int begin = 0;
int end = 0;

void rising(){
    begin = tim.read_us();
}

void falling(){
    end = tim.read_us();
}

int main(void){
    float distance = 0;
    
    trig.period_ms(60);     // period      = 60ms
    trig.pulsewidth_us(10); // pulse-width = 10us
    
    echo.rise(&rising);
    echo.fall(&falling);
    
    tim.start();
    
    while(1){
        distance =  (float)(end - begin) / 58; // [cm]
        pc.printf("Distance = %.2f[cm]\r\n", distance);
        wait(0.5);
    }
    
} 


```
{% endtab %}

{% tab title="EC" %}

{% endtab %}

{% tab title="Arduino" %}

{% endtab %}
{% endtabs %}

### 

## Timer

{% tabs %}
{% tab title="mbed" %}
```cpp

#include "mbed.h"

Timer       timer;
Serial      pc(USBTX, USBRX, 9600); // for using ‘printf()’

int begin, end;
int cnt = 0;

int main(void){

    timer.start();
    
    begin = timer.read_us();
    
    while(cnt < 100) cnt++;
    
    end = timer.read_us();
    
    pc.printf("Counting 100 takes %d [us]", end-begin);
}
```
{% endtab %}

{% tab title="EC" %}
```

```
{% endtab %}

{% tab title="Arduino" %}

{% endtab %}
{% endtabs %}

### 

## Input Capture

{% tabs %}
{% tab title="mbed" %}
```cpp
#include "mbed.h"

Serial      pc(USBTX, USBRX, 9600);
PwmOut      trig(D10); // Trigger 핀
InterruptIn echo(D7);  // Echo 핀
Timer       tim;

int begin = 0;
int end = 0;

void rising(){
    begin = tim.read_us();
}

void falling(){
    end = tim.read_us();
}

int main(void){
    float distance = 0;
    
    trig.period_ms(60);     // period      = 60ms
    trig.pulsewidth_us(10); // pulse-width = 10us
    
    echo.rise(&rising);
    echo.fall(&falling);
    
    tim.start();
    
    while(1){
        distance =  (float)(end - begin) / 58; // [cm]
        pc.printf("Distance = %.2f[cm]\r\n", distance);
        wait(0.5);
    }
    
} 



```
{% endtab %}

{% tab title="EC" %}

{% endtab %}

{% tab title="Arduino" %}

{% endtab %}
{% endtabs %}

### 

## Ticker

{% tabs %}
{% tab title="mbed" %}
```cpp
#include "mbed.h"

Ticker     tick;
DigitalOut led(LED1);

void INT(){
    led = !led;      
}
int main(void){
    tick.attach(&INT, 1); // 1초마다 LED blink
    
    while(1);
}

```
{% endtab %}

{% tab title="EC" %}

{% endtab %}

{% tab title="Arduino" %}

{% endtab %}
{% endtabs %}

### 

## ADC

{% tabs %}
{% tab title="mbed" %}
```cpp
#include "mbed.h"
                                                
Serial      pc(USBTX, USBRX, 9600);                                                
AnalogIn    CDS(A0);
DigitalOut  led(LED1);

int main() {
    float measure;
        
    while(1) {
        measure = CDS.read(); // mapping(0~3.3V -> 0.0~1.0)
        measure = measure * 3300; // [mV] (0.0~1.0 -> 0~3300[mV])
        pc.printf("measure = %f mV\n\r", measure);
        
        if (measure < 200) led = 1;
        else               led = 0;
        
        wait(0.2); 
    }
}

```
{% endtab %}

{% tab title="EC" %}

{% endtab %}

{% tab title="Arduino" %}

{% endtab %}
{% endtabs %}

## UART

{% tabs %}
{% tab title="mbed" %}
```cpp
#include "mbed.h"
 
Serial  uart(USBTX, USBRX, 9600);
 
int main(){
    char RXD;    
    while(1)
    {        
        if(uart.readable()){
            RXD = uart.getc();
            uart.printf("%c", RXD);
        }
    }
}

```
{% endtab %}

{% tab title="EC" %}

{% endtab %}

{% tab title="Arduino" %}

{% endtab %}
{% endtabs %}

### 

## 

