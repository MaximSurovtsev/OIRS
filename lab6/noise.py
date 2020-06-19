import random
from PIL import Image, ImageDraw #Подключим необходимые библиотеки. 
from imutils import paths
import os


TEST_IMAGES = 'data/train'
test_images = list(paths.list_images(TEST_IMAGES))
print(test_images)
for path in test_images[:10]:
	image = Image.open(path) #Открываем изображение. 
	draw = ImageDraw.Draw(image) #Создаем инструмент для рисования. 
	width = image.size[0] #Определяем ширину. 
	height = image.size[1] #Определяем высоту. 	
	pix = image.load() #Выгружаем значения пикселей.

	for i in range(width):
		for j in range(height):
			d = pix[i, j][3] 
			if d == 0:
				a = b = c = 255
			else:
				a = pix[i, j][0] + random.randint(-100, 100)
				b = pix[i, j][1] + random.randint(-100, 100)
				c = pix[i, j][2] + random.randint(-100, 100)
				

			
				
			draw.point((i, j), (a, b, c))
	image.save(f'data/test/{random.randint(2**32, 2**64)}.png', "PNG")
	del draw