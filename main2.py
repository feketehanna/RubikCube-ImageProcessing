# FONTOS: A képek sorrendje a hálóhoz:
# 1.jpg --> Felső 
# 2.jpg --> Bal 
# 3.jpg --> Első  
# 4.jpg --> Jobb 
# 5.jpg --> Hátsó 
# 6.jpg --> Alsó 


import cv2
import numpy as np


#--------------------------szín felismerés logika--------------------------

def get_color_bgr(hsv_pixel):

    # h - színárnyalat (0-179), s - telítettség(0-255), v - fényerő(0-255)
    h, s, v = hsv_pixel

    # Fehér (alacsony telítettség, magas fényerő)
    if s < 50 and v > 150: return (255, 255, 255)

    # Piros (két tartománya van a kör végén)
    if h < 10 or h > 165: return (0, 0, 255)
    # Narancs
    if 10 <= h < 22: return (0, 165, 255)
    # Citromsarga
    if 22 <= h < 45: return  (0, 255, 255)
    # Zold 
    if 45 <= h < 90: return (0, 255, 0)
    # Kek
    if 90 <= h < 130: return (255, 0, 0)

    #ha nem találja az értéket szürke színnel tér majd vissza 
    return (128, 128, 128)

#------------------------------------------------------------------------------


#--------------------------egy oldal feldolgozása--------------------------
def process_face(filename): #előre definiált függvény a kép beolvasására 
    img = cv2.imread(filename) #fájlnév beolvasása 

    #------- hibakezelés ha nem található a kép --------
    if img is None:
        print(f"HIBA: A {filename} nem található!")
        return None


    #------- előfeldolgozás az élekhez--------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    blur = cv2.GaussianBlur(gray, (5, 5), 0) #kép szürkéssé tevése --> világos-sötét külonbségek

    #edges --> kimenet fekete kép, fehér vonalakkal 
    edges = cv2.Canny(blur, 30, 100) #kicsit elhomályosítja a képet --> gép ne lássa a zavaró tényezőket 
 

    # Szín kinyeréséhez HSV kell
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # színes kép készítése --> ebből szedjük majd ki a színeket  
    
    # fehér körvonalak menten korbe jarja az alakzatokat --> kimenet egy lista lesz az összes talált formával (hatterben levo dolgok is)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    #------- szures --------
    stickers = []

    for cnt in contours: #végig iterálunk a talált fomákon
        area = cv2.contourArea(cnt) #tul nagy dolgok kidobása --> csak a matricák maradnak a kockárol

        if 1000 < area < 100000:

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True) #görbe vonalak sarkítása --> négy sarku alakzat végeredményként 

            if 4 <= len(approx) <= 6: # csak azok maradnak bent aminket negy, ot vagy hat sarka van 
                #alakzat sulypontjának kíszámítása 
                M = cv2.moments(cnt)

                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) # éppen adott alakzat x koordinátája 
                    cy = int(M["m01"] / M["m00"]) # éppen adott alakzat y koordinátája 
                    color = get_color_bgr(hsv[cy, cx]) # matrica közepérol kinyerjuk a színét --> meeghívjuk a megírt fuggyvenyt 
                    stickers.append({'pos': (cx, cy), 'color': color}) # listába elmentjuk a helyet es a szint is   
    
    #------- hibakezeles --------
    if len(stickers) < 9: 
        print(f"FIGYELEM: A {filename} képen csak {len(stickers)} matricát találtam!")
        # ha nincs meg 9 --> feltöltjük feketével, hogy ne omoljon össze
        while len(stickers) < 9:
            stickers.append({'pos': (0,0), 'color': (0,0,0)})


    #------- rácsba rendezés (fentről le, balról jobbra) --------
    stickers.sort(key=lambda m: m['pos'][1]) #sorba rakja oket magassag szerint (melyik van kozepen, felul es alul)
    for i in range(0, 9, 3):
        row = stickers[i:i+3]
        row.sort(key=lambda m: m['pos'][0]) # sorokon belul sorba rakja oket 
        stickers[i:i+3] = row 
    
    #visszaadja a kilenc szint egy egyszeru listaban 
    return [s['color'] for s in stickers[:9]]

#------------------------------------------------------------------------------


#--------------------------főprogram--------------------------
def main():
   
   
    # Sorrend: Felső, Bal, Első, Jobb, Hátsó, Alsó
    filenames = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg'] #kepek betoltese egy listaba 
    all_faces = [] #eredmény lista -- 54 db színkód sorban 

    print("Képek feldolgozása folyamatban .-.'")
    #képrek megnyitása egyesével és feldolgozása a megírt függvény segítségével  
    for f in filenames:
        colors = process_face(f)
        all_faces.append(colors)

    # Háló kirajzolása
    m = 40    # egy kismatrica mérete pixelben 
    side = 3 * m     # egy oldal mérete 
    canvas = np.zeros((side * 3, side * 4, 3), dtype=np.uint8) #teljesen üres fekete kép létrehozasa 
    #3 oldalnyi a magassága és négy oldalnyi a szélessége 

    # A kereszt alakzat elrendezése (Sor, Oszlop)
    offsets = [(0,1), (1,0), (1,1), (1,2), (1,3), (2,1)] #koordináta rendszer megadása 

    #végigmegyünk a hat oldalon és azok helyein (matricáin)
    for face_idx, (r_off, c_off) in enumerate(offsets): 
        face_colors = all_faces[face_idx] #adott oldalhoz tartozó 9 szín kivétele 
        #iterálás egy oldal matricáin 
        for i in range(9):
            row, col = i // 3, i % 3 # // -- melyik sor, % -- melyik oszlop 
            #elhelyezés a koordináta rendszerben 
            x = (c_off * side) + (col * m) 
            y = (r_off * side) + (row * m)
            #négyzet megrajzolása az adott pozicióba , (+2, -2 -- kicsike fekete hely a matricák között)
            cv2.rectangle(canvas, (x+2, y+2), (x+m-2, y+m-2), face_colors[i], -1) # -1 -- kitöltés teljesen 

    print("Kész! A háló megjelent a képernyőn. :33")
    cv2.imshow('Rubik kocka haloja *-*', canvas) #feldobja az ablakot és a kész hálót 
    cv2.waitKey(0) #megállítja a programot és vár amíg be nem zárod az ablakot 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()