from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from ..core_threading import MulDeviceSynCapture
from threading import Thread


class Demo(object):
    '''同步demo'''
    def __init__(self):
        # 初始化窗口
        self.root = tk.Tk()
        self.root.title('Demo')
        h, w = 400, 1000
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f'{w}x{h}+{x}+{y}')
        self.start_flag = True
        
    def init(self):
        self.image_labels = []
        self.mc = MulDeviceSynCapture(0, [1, 2])
        
    def change_flag(self):
        self.start_flag = not self.start_flag
        self.control_button.config(text='start' if not self.start_flag else 'stop')
    
    def create_ui(self):
        '''构造ui'''
        self.control_button = ttk.Button(self.root, text="stop", 
                                        command=self.change_flag, 
                                        width=5)
        self.control_button.pack(pady=20)
        self.image_frame = ttk.Frame()
        self.image_frame.pack()
        tk_image_list = self.get_syc_image()
        for i, tk_image in enumerate(tk_image_list):
            image_label = ttk.Label(self.image_frame, name=f'device: {i}', image=tk_image)
            image_label.pack(side=tk.LEFT, padx=10)
            self.image_labels.append(image_label)
        
    def run(self):
        self.init()
        # 构造ui
        self.create_ui()
        # 设置退出是操作
        self.root.protocol('WM_DELETE_WINDOW', self.before_quit)
        # 开启显示同步线程
        thread = Thread(target=self.image_loop)
        thread.setDaemon(True)
        thread.start()
        # 开启事件循环
        self.root.mainloop()
        
    def get_syc_image(self):
        ret = []
        for data in self.mc.get():
            image = Image.fromarray(data[1])
            # 缩放图片  Image.LANCZOS
            image = image.resize((320, 240), Image.Resampling.LANCZOS)
            # 把PIL图像对象转变为Tkinter的PhotoImage对象
            tk_image = ImageTk.PhotoImage(image)
            ret.append(tk_image)
        return ret

    def before_quit(self):
        self.start_flag = False
        # 刷新输入并关闭文件
        if self.mc:
            self.mc.close()
        # 关闭窗口
        self.root.destroy()

    def image_loop(self):
        '''保存并改变图片'''
        while True:
            # 改变图片
            # 获取image_label
            if not self.start_flag:
                continue
            for image_label, tk_image in zip(self.image_labels, self.get_syc_image()):
                # # 修改图片
                image_label.config(image=tk_image)
                image_label.image = tk_image

def run():
    print(f'设置环境变量 FAKE=1 使用模拟环境')
    getimage = Demo()
    getimage.run()

if __name__ == '__main__':
    run()
