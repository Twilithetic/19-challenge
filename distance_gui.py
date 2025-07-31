import tkinter as tk
from tkinter import ttk
from multiprocessing import Process, Queue
import time
from distance_service import distance_calculation_service

class DistanceMeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("target measure ")
        self.root.geometry("1024x600")
        
        # ???????????
        self.result_queue = Queue()
        self.control_queue = Queue()
        
        # ??????????
        self.distance_process = Process(
            target=distance_calculation_service,
            args=(self.result_queue, self.control_queue)
        )
        self.distance_process.daemon = True
        self.distance_process.start()
        
        # ????
        self.running = False
        self.current_distance = "?????"
        
        # ??UI??
        self.create_widgets()
        
        # ??????????
        self.update_distance()
    
    def create_widgets(self):
        # ?????
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ??
        title_label = ttk.Label(
            main_frame, 
            text="A4???????", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # ????
        self.status_var = tk.StringVar(value="??: ??")
        status_label = ttk.Label(
            main_frame, 
            textvariable=self.status_var, 
            font=("Arial", 12)
        )
        status_label.pack(pady=5)
        
        # ????
        self.distance_var = tk.StringVar(value="??: ???")
        distance_label = ttk.Label(
            main_frame, 
            textvariable=self.distance_var, 
            font=("Arial", 24, "bold"),
            foreground="#007ACC"
        )
        distance_label.pack(pady=20)
        
        # ??/????
        self.start_button = ttk.Button(
            main_frame,
            text="????",
            command=self.toggle_measurement,
            width=15
        )
        self.start_button.pack(pady=20)
        
        # ????
        exit_button = ttk.Button(
            main_frame,
            text="??",
            command=self.on_exit,
            width=10
        )
        exit_button.pack(pady=10)
    
    def toggle_measurement(self):
        """?????????/???"""
        if self.running:
            # ????
            self.control_queue.put("STOP")
            self.running = False
            self.start_button.config(text="????")
            self.status_var.set("??: ???")
        else:
            # ????
            self.control_queue.put("START")
            self.running = True
            self.start_button.config(text="????")
            self.status_var.set("??: ???")
    
    def update_distance(self):
        """???????????????UI"""
        if not self.result_queue.empty():
            distance = self.result_queue.get()
            if distance is not None:
                self.current_distance = f"{distance:.1f} cm"
            else:
                self.current_distance = "???"
            
            self.distance_var.set(f"??: {self.current_distance}")
        
        # ?100??????
        self.root.after(100, self.update_distance)
    
    def on_exit(self):
        """??????"""
        # ???????????
        self.control_queue.put("EXIT")
        
        # ?????????1??
        self.distance_process.join(1.0)
        
        # ????????????
        if self.distance_process.is_alive():
            self.distance_process.terminate()
        
        # ????
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DistanceMeasurementApp(root)
    root.mainloop()    