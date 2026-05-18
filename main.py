# main.py
import tkinter as tk
import os
from tkinter import filedialog, messagebox
from image import Image
from segment import segment
from sample_processing import adaptive_size_gating
from data import write_to_xlsx


class GroupingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TIFF Grouping Tool")
        self.root.geometry("900x500")

        self.tif_files = []
        self.image_directory = ''
        self.groups = {}

        # --- LEFT SIDE: FILE LIST ---
        left_frame = tk.Frame(root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(left_frame, text="Available TIFF Files").pack()

        self.file_listbox = tk.Listbox(
            left_frame,
            selectmode=tk.MULTIPLE,
            width=50,
            height=25
        )
        self.file_listbox.pack(fill=tk.BOTH, expand=True)

        # --- RIGHT SIDE: GROUPS ---
        right_frame = tk.Frame(root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(right_frame, text="Groups").pack()

        self.group_listbox = tk.Listbox(
            right_frame,
            width=40,
            height=20
        )
        self.group_listbox.pack(fill=tk.BOTH, expand=True)

        # Group controls
        controls_frame = tk.Frame(right_frame)
        controls_frame.pack(fill=tk.X, pady=10)

        tk.Label(controls_frame, text="Group Name:").pack(side=tk.LEFT)

        self.group_name_entry = tk.Entry(controls_frame)
        self.group_name_entry.pack(side=tk.LEFT, padx=5)

        add_button = tk.Button(
            controls_frame,
            text="Create Group",
            command=self.create_group
        )
        add_button.pack(side=tk.LEFT, padx=5)

        # RUN button
        run_button = tk.Button(
            right_frame,
            text="RUN",
            height=2,
            bg="green",
            fg="white",
            command=self.run_processing
        )
        run_button.pack(fill=tk.X, pady=10)

        # Ask user for directory at startup
        self.select_directory()

    def select_directory(self):
        #self.image_directory = filedialog.askdirectory(title="Select Directory Containing TIFF Files")
        self.image_directory = './images/'

        if not self.image_directory:
            self.root.destroy()
            return

        tif_extensions = (".tif", ".tiff")

        self.tif_files = sorted([
            f for f in os.listdir(self.image_directory)
            if f.lower().endswith(tif_extensions)
        ])

        if not self.tif_files:
            messagebox.showwarning("No TIFF Files", "No .tif or .tiff files found.")
            return

        self.file_listbox.delete(0, tk.END)

        for file in self.tif_files:
            self.file_listbox.insert(tk.END, file)

    def create_group(self):
        group_name = self.group_name_entry.get().strip()

        if not group_name:
            messagebox.showwarning("Missing Name", "Please enter a group name.")
            return

        selected_indices = self.file_listbox.curselection()

        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select TIFF files.")
            return

        selected_files = [self.file_listbox.get(i) for i in selected_indices]

        # Ensure exclusivity:
        # Remove selected files from existing groups
        for group in self.groups.values():
            for f in selected_files:
                if f in group:
                    group.remove(f)

        # Add/update group
        self.groups[group_name] = selected_files

        self.refresh_group_display()

        # Clear selection
        self.file_listbox.selection_clear(0, tk.END)
        self.group_name_entry.delete(0, tk.END)

    def refresh_group_display(self):
        self.group_listbox.delete(0, tk.END)

        for group_name, files in self.groups.items():
            self.group_listbox.insert(
                tk.END,
                f"{group_name}: {len(files)} files"
            )

            for f in files:
                self.group_listbox.insert(
                    tk.END,
                    f"    - {f}"
                )

    def run_processing(self):
        """
        Placeholder processing function.
        Receives the grouping dictionary.
        """

        self.process_groups(self.groups)

        # Create result window
        result_window = tk.Toplevel(self.root)
        result_window.title("Processing Finished")
        result_window.geometry("400x200")

        label = tk.Label(
            result_window,
            text="Processing completed successfully!",
            font=("Arial", 14)
        )
        label.pack(expand=True, pady=40)


    def process_groups(self, groups):
        """
        Placeholder function that receives the grouping structure.

        Example structure:
        {
            "Group A": ["file1.tif", "file2.tif"],
            "Group B": ["file3.tif"]
        }
        """

        print("Received groups:")
        sample_groups = dict()
        for group_name, files in groups.items():

            # Print debug information
            print(group_name, "->", [os.path.join(self.image_directory, f) for f in files])

            # Create an image object, analyze and subsequently delete image file from memory.
            for file in files:
                image_path = os.path.join(self.image_directory, file)
                image_object = Image(file, image_path)

                segment([image_object])

                image_object.clear_img()

                if group_name not in sample_groups:
                    sample_groups[group_name] = [image_object]
                else:
                    sample_groups[group_name].append(image_object)

        adaptive_size_gating(sample_groups)

        accumulated_groups = {k: k for k in self.groups.keys()}
        current_accumulated_groups = accumulated_groups.keys()

        write_to_xlsx(
            os.path.join(self.image_directory, 'output'),
            current_accumulated_groups,
            accumulated_groups,
            sample_groups,
            (500, 500),
            0.1624808538555503
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = GroupingApp(root)
    root.mainloop()