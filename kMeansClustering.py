# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:12:28 2021

@author: VolkanKarakuş
"""

# suana kadar supervised learning kismini bitirdik. Elimizde labellar vardi, datalarin hangi label'a ait olduguna bakiyorduk.
# Suanki konu unsupervised learning. Bu kisimda elimizde label yok.
# Biz K= 2,3,5 icin bu kadar label var olarak kabul edip, datalari gruplicaz.
# clustering euclidian distance ile yapilir.
# k=2 icin iki tane rastgele centroid alinip, diger datalarin uzakligina göre bir merkez alinir.
# bu islem tekrarlandiginda veriler cluster edilmis olur.

# biz peki bu islemi dogru mu yapiyoruz ?
# mesela ben k=4 icin cluster ettim ama aslinda elimde 2 tane label vardi. 
# bunun dogru olup olmadigi WCSS ile bulunur.
# WCSS : centroid merkezinin cluster icindeki noktalar euclidian distance'inin kareleri toplami.
# biz WCSS 'yi azaltmak istiyoruz ama elimde 1000 tane data varsa 1000 tane de cluster yapmaya calismayacagim.
# K icin grafige bakip kirilma noktasi optimumdur denebilir.