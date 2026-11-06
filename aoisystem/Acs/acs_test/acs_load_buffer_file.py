import os
import SPiiPlusPython as sp

def run_buffer_from_file(conn, buf_num, filename):
    """
    将文件内容加载到指定 buffer，编译并执行，
    并使用 WaitProgramEnd 等待程序执行结束
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件 {filename} 不存在。")
    
    with open(filename, "r") as f:
        code = f.read()

    # 1. 加载程序到指定缓冲区（会先清空该缓冲区）
    sp.LoadBuffer(conn, buf_num, code, len(code), sp.SYNCHRONOUS, True)
    
    # 2. 编译缓冲区程序
    sp.CompileBuffer(conn, buf_num, sp.SYNCHRONOUS, True)
    
    # 3. 运行缓冲区程序（从第一行开始执行，故 label 为 None）
    sp.RunBuffer(conn, buf_num, None, sp.SYNCHRONOUS, True)

    # result = sp.GetResult(conn, sp.SYNCHRONOUS)
   # 4. 使用 WaitProgramEnd 等待程序执行结束，等待超时设置为 30000 毫秒（30 秒）
    sp.WaitProgramEnd(conn, buf_num, 180000, True)
    
    # return result


def main():
    # 建立与控制器的通信连接，请根据实际情况修改 IP 和端口
    ip = "10.0.0.100"
    port = 701
    conn = sp.OpenCommEthernetTCP(ip, port)
    if conn == -1:
        print("连接控制器失败，请检查 IP、端口或网络。")
        return
    
    try:
        # 依次运行 buffer_1.txt, buffer_2.txt, buffer_3.txt（文件名请确认正确）
        run_buffer_from_file(conn, 1, "/home/aoi/pretest/AOI_V1/Acs/acs_test/Buffers_use/buffer_x_home.txt")
        run_buffer_from_file(conn, 2, "/home/aoi/pretest/AOI_V1/Acs/acs_test/Buffers_use/buffer_y_home.txt")                                          
        run_buffer_from_file(conn, 3, "/home/aoi/pretest/AOI_V1/Acs/acs_test/Buffers_use/buffer_x_init_100.txt")

        run_buffer_from_file(conn, 4, "/home/aoi/pretest/AOI_V1/Acs/acs_test/Buffers_use/buffer_y_forward.txt")

        run_buffer_from_file(conn, 5, "/home/aoi/pretest/AOI_V1/Acs/acs_test/Buffers_use/buffer_x_100_negative.txt")

        run_buffer_from_file(conn, 6, "/home/aoi/pretest/AOI_V1/Acs/acs_test/Buffers_use/buffer_y_back.txt")

        # run_buffer_from_file(conn, 5, "Buffers_use/buffer_x_50_negative.txt")
        # run_buffer_from_file(conn, 4, "Buffers_use/buffer_y_forward.txt")
        # run_buffer_from_file(conn, 5, "Buffers_use/buffer_x_50_negative.txt")
        # run_buffer_from_file(conn, 6, "Buffers_use/buffer_y_back.txt")        
        print("you have complished full PCB collection!")


    finally:
        sp.DisableAll()
        # 关闭与控制器的连接
        sp.CloseComm(conn)

if __name__ == "__main__":
    main()
