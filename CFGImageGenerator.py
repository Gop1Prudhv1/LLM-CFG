from graphviz import Source
from PIL import Image
from io import BytesIO
import re


class CFGImageGenerator:
    def generate_image(self, dot_string):
        # Render the DOT string into a PNG image
        dot_string = re.sub(r'\\"', '', dot_string)
        dot_string = re.sub(r'```dot', '', dot_string)
        dot_string = re.sub(r'```', '', dot_string)

        print('*** printing the fixed dot string ****')
        print(dot_string)
        src = Source(dot_string, format="png")
        png_bytes = src.pipe()

        # Display the PNG image
        img = Image.open(BytesIO(png_bytes))
        img.show()