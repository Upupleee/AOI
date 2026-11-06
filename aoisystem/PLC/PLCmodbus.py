from pymodbus.client import ModbusTcpClient
from pymodbus.payload import BinaryPayloadDecoder, BinaryPayloadBuilder
from pymodbus.constants import Endian
import time
import struct
# PLC 连接类
class PLCConnection:
    def __init__(self, ip, port=502):
        self.ip = ip
        self.port = port
        self.client = ModbusTcpClient(self.ip, port=self.port)
        self.connect()

    def connect(self):
        while not self.client.connect():
            print(f"Connection error to {self.ip}:{self.port}. Retrying in 5 seconds...")
            time.sleep(5)
        print(f"Connected to PLC at {self.ip}:{self.port}")
        
    def disconnect(self):
        self.client.close()

    def reconnect(self):
        self.client.close()
        self.connect()

    def read_bits(self, address, count):
        try:
            result = self.client.read_coils(address, count)
            if result.isError():
                print(f"Read bits error: {result}")
                self.reconnect()
                return self.read_bits(address, count)
            return result.bits
        except Exception as e:
            print(f"Read bits error: {e}. Reconnecting...")
            self.reconnect()
            return self.read_bits(address, count)

    def write_bit(self, address, value):
        try:
            result = self.client.write_coil(address, value)
            if result.isError():
                print(f"Write bit error: {result}")
                self.reconnect()
                self.write_bit(address, value)
        except Exception as e:
            print(f"Write bit error: {e}. Reconnecting...")
            self.reconnect()
            self.write_bit(address, value)

    def read_registers(self, address, count):
        try:
            result = self.client.read_holding_registers(address, count, slave= 1)
            if result.isError():
                print(f"Read registers error: {result}")
                self.reconnect()
                return self.read_registers(address, count)
            return result.registers
        except Exception as e:
            print(f"Read registers error: {e}. Reconnecting...")
            # self.reconnect()
            # return self.read_registers(address, count)

    def write_register(self, address, value):
        try:
            result = self.client.write_register(address, value)
            if result.isError():
                print(f"Write register error: {result}")
                self.reconnect()
                self.write_register(address, value)
        except Exception as e:
            print(f"Write register error: {e}. Reconnecting...")
            # self.reconnect()
            # self.write_register(address, value)
    
    def read_float(self, address):
        registers = self.read_registers(address, 2)
        decoder = BinaryPayloadDecoder.fromRegisters(registers, byteorder=Endian.BIG, wordorder=Endian.LITTLE)
        return decoder.decode_32bit_float()

    def write_float(self, address, value):
        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.LITTLE)
        builder.add_32bit_float(value)
        registers = builder.to_registers()
        try:
            result = self.client.write_registers(address, registers)
            if result.isError():
                print(f"Write float error: {result}")
                self.reconnect()
                self.write_float(address, value)
        except Exception as e:
            print(f"Write float error: {e}. Reconnecting...")
            self.reconnect()
            self.write_float(address, value)


if __name__ == "__main__":
    from common.config import config
    plc = PLCConnection(config.plc_ip, port=config.plc_port)
    ready_status = plc.read_bits(config.coil('alarm_reset'), 1)[0]
    print(ready_status)
    plc.disconnect()