from graphviz import Source
from PIL import Image
from io import BytesIO


class CFGImageGenerator:
    def generate_image(self, dot_string):
        # Render the DOT string into a PNG image
        src = Source(dot_string, format="png")
        png_bytes = src.pipe()

        # Display the PNG image
        img = Image.open(BytesIO(png_bytes))
        img.show()