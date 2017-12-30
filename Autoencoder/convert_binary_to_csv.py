import numpy as np
import struct

'''
data0 is binary file so we convert it to float or integer
File format:
Each file has 1000 training examples. Each training example is of size 28x28 pixels.
The pixels are stored as unsigned chars (1 byte) and take values from 0 to 255.
The first 28x28 bytes of the file correspond to the first training example,
the next 28x28 bytes correspond to the next example and so on.

2^2 = 4
2^4 = 16
2^8 = 256
2^16 = 65536

// 1 byte -> [0-255] or [0x00-0xFF]
uint8_t         number8     = testValue; // 255
unsigned char    numberChar    = testValue; // 255

// 2 bytes -> [0-65535] or [0x0000-0xFFFF]
uint16_t         number16     = testValue; // 65535
unsigned short    numberShort    = testValue; // 65535

// 4 bytes -> [0-4294967295] or [0x00000000-0xFFFFFFFF]
uint32_t         number32     = testValue; // 4294967295
unsigned int     numberInt    = testValue; // 4294967295

 // 8 bytes -> [0-18446744073709551615] or [0x0000000000000000-0xFFFFFFFFFFFFFFFF]
uint64_t             number64         = testValue; // 18446744073709551615
unsigned long long     numberLongLong    = testValue; // 18446744073709551615

'''
fin = open("data8", "rb")
a = np.fromfile(fin, dtype=np.uint8)
print('The traing examples 1000 = ', a.shape[0]/(28*28))
print('Size = ', a.shape)
a = a.reshape(1000, 784)
print('Size = ', a.shape)

# save
np.savetxt('data8.csv', a, delimiter=',')

