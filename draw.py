import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
img = plt.imread("taiwan.png")
fig, ax = plt.subplots()
ax.imshow(img)
x = [160, 150, 140, 110, 95, 100]
y = [40, 43, 45, 90, 140, 170]
notation = ["Guting", "Banquio", "Zhongli", "Xitun", "Tainan", "Qianzhen"]
# ax.imshow(img, extent=[0, 400, 0, 300])
ax.scatter(x, y)
for i, txt in enumerate(notation):
    ax.annotate(txt, (x[i], y[i]))
plt.show()
