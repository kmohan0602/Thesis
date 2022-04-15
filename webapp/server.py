import socket
import numpy as np
import encodings

HOST = '127.0.0.1'
PORT = 65432


def my_server():
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("Server Started waiting for client to connect")
        s.bind((HOST, PORT))
        s.listen(5)
        conn, addr = s.accept()

        print(conn)
        print(addr)
        data = "Hello"
        
        with conn:
            print('Connected by', addr)
            while True:

                data = conn.recv(1024).decode('utf-8')
                print(data)
                if str(data)=="Data":
                    print("Ok Sending Data")

                    x = np.random.randint(0,100,None)
                    x = str(x)
                    x_encoded = x.encode('utf-8')

                    conn.sendall(x_encoded)

                elif str(data) == "Quit":
                    print("Shutting Server")
                    break
                    
                if not data:
                    break
                else:
                    pass
    return data


if __name__ == '__main__':
    while(1):
        print('herre')
        temp = my_server()
        print(temp)

# if __name__ == '__main__':
#     my_server()