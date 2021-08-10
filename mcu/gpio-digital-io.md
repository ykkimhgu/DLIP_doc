# GPIO Digital IO



```cpp
////    -2021-      //////
class DigitalIn
{
public:
    DigitalIn(PinName pin) : gpio()
    {
        gpio_init_in(&gpio, pin);
        //gpio() 교체하기!
    }

    DigitalIn(PinName pin, PinMode mode) : gpio()
    {
        gpio_init_in_ex(&gpio, pin, mode);
    }

    ~DigitalIn()
    {
        gpio_free(&gpio);
    }

    int read()
    {
        return gpio_read(&gpio);
        //GPIO_Read(GPIOX, Pin, IDR)
    }

    void mode(PinMode pull);
    int is_connected()
    {
        return gpio_is_connected(&gpio);
    }

    operator int()
    {
        return read();
    }

protected:
#if !defined(DOXYGEN_ONLY)
    gpio_t gpio;
#endif //!defined(DOXYGEN_ONLY)
};

```



```cpp
///////////////////////////////////////////////////////
// ec C functions for LAB

void GPIO_init(GPIO_TypeDef* Port, int Pin, int mode);
void GPIO_write(GPIO_TypeDef* Port, int Pin, int Output);
int  GPIO_read(GPIO_TypeDef* Port, int Pin);
void GPIO_Initialize(GPIOX, Pin, I / OMode)
void GPIO_OutMode(GPIOX, Pin, Outmode)
void GPIO_OutPUDR(GPIOX, Pin, PUDR)
void GPIO_InMode(GPIOX, Pin, InMode)
void GPIO_InPUDR(GPIOX, Pin, PUDR)
void GPIO_Read(GPIOX, Pin, IDR)
void GPIO_Write(GPIOX, Pin, ODR)
```

