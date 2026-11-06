import SPiiPlusPython as sp

def test_communication():
    # 定义控制器的 IP 地址和端口
    ip = "10.0.0.100"
    port = 701

    # 建立以太网 TCP 连接
    hc = sp.OpenCommEthernetTCP(ip, port)
    if hc == -1:
        print("无法连接到控制器。")
        return
    cmd = "?VR\r"
    response = sp.Transaction(hc, cmd, len(cmd), 256, sp.SYNCHRONOUS, failure_check=True)
    print("控制器响应:", response)

    # 关闭通信通道，释放资源
    sp.CloseComm(hc)

if __name__ == "__main__":
    test_communication()
