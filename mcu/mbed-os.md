# mbed OS

##  API List 

[Full API list for mbed](https://os.mbed.com/docs/mbed-os/v6.13/apis/index.html)



#### Serial \(UART\) drivers

| API | Full profile | Bare metal profile |
| :--- | :--- | :--- |
| [BufferedSerial](https://os.mbed.com/docs/mbed-os/v6.13/apis/serial-uart-apis.html) | ✔ | ✔ |
| [UnbufferedSerial](https://os.mbed.com/docs/mbed-os/v6.13/apis/unbufferedserial.html) | ✔ | ✔ |

#### SPI drivers

| API | Full profile | Bare metal profile |
| :--- | :--- | :--- |
| [QuadSPI \(QSPI\)](https://os.mbed.com/docs/mbed-os/v6.13/apis/spi-apis.html) | ✔ | ✔ |
| [SPI](https://os.mbed.com/docs/mbed-os/v6.13/apis/spi.html) | ✔ | ✔ |
| [SPISlave](https://os.mbed.com/docs/mbed-os/v6.13/apis/spislave.html) | ✔ | ✔ |

#### Input/Output drivers

| API | Full profile | Bare metal profile |
| :--- | :--- | :--- |
| [AnalogIn](https://os.mbed.com/docs/mbed-os/v6.13/apis/i-o-apis.html) | ✔ | ✔ |
| [AnalogOut](https://os.mbed.com/docs/mbed-os/v6.13/apis/analogout.html) | ✔ | ✔ |
| [BusIn](https://os.mbed.com/docs/mbed-os/v6.13/apis/busin.html) | ✔ | ✔ |
| [BusOut](https://os.mbed.com/docs/mbed-os/v6.13/apis/busout.html) | ✔ | ✔ |
| [BusInOut](https://os.mbed.com/docs/mbed-os/v6.13/apis/businout.html) | ✔ | ✔ |
| [DigitalIn](https://os.mbed.com/docs/mbed-os/v6.13/apis/digitalin.html) | ✔ | ✔ |
| [DigitalOut](https://os.mbed.com/docs/mbed-os/v6.13/apis/digitalout.html) | ✔ | ✔ |
| [DigitalInOut](https://os.mbed.com/docs/mbed-os/v6.13/apis/digitalinout.html) | ✔ | ✔ |
| [InterruptIn](https://os.mbed.com/docs/mbed-os/v6.13/apis/interruptin.html) | ✔ | ✔ |
| [PortIn](https://os.mbed.com/docs/mbed-os/v6.13/apis/portin.html) | ✔ | ✔ |
| [PortOut](https://os.mbed.com/docs/mbed-os/v6.13/apis/portout.html) | ✔ | ✔ |
| [PortInOut](https://os.mbed.com/docs/mbed-os/v6.13/apis/portinout.html) | ✔ | ✔ |
| [PwmOut](https://os.mbed.com/docs/mbed-os/v6.13/apis/pwmout.html) | ✔ | ✔ |



## \# Library 

header: [github](https://github.com/ARMmbed/mbed-os/tree/master/drivers/include/drivers)

source:[github](https://github.com/ARMmbed/mbed-os/tree/master/drivers/source)

[Intenal Library](https://os.mbed.com/handbook/mbed-library-internals.)

## DigitalIn

```cpp
class DigitalIn
{
    public:
        DigitalIn(PinName pin) : gpio()
        {
            gpio_init_in(&gpio, pin);
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
void DigitalIn::mode(PinMode pull)
{
    core_util_critical_section_enter();
    gpio_mode(&gpio, pull);
    core_util_critical_section_exit();
}
```

### DigitalOut

```cpp
class DigitalOut
{
public:
    DigitalOut(PinName pin) : gpio()
    {
        gpio_init_out(&gpio, pin);
    }

    DigitalOut(PinName pin, int value) : gpio()
    {
        gpio_init_out_ex(&gpio, pin, value);
    }

    void write(int value)
    {
        gpio_write(&gpio, value);
    }

    int read()
    {
        return gpio_read(&gpio);
    }

    int is_connected()
    {
        return gpio_is_connected(&gpio);
    }

    DigitalOut& operator= (int value)
    {
        write(value);
        return *this;
    }

    DigitalOut& operator= (DigitalOut& rhs);

    operator int()
    {
        return read();
    }

};
```

