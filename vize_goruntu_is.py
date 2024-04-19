import cv2
import numpy as np
import time

def main(video_dosyasi):
    cap = cv2.VideoCapture("C:\\Users\\Lenovo\\Desktop\\vid_2.avi")
    if not cap.isOpened():
        print("Video acilamiyor.")
        return

    # Kırmızı ve beyaz toplar için HSV eşik değerleri
    kirmizi_alt = np.array([160, 100, 100])
    kirmizi_ust = np.array([180, 255, 255])
    beyaz_alt = np.array([0, 0, 168])
    beyaz_ust = np.array([172, 111, 255])

    onceki_kirmizi_merkez = None
    hareket_izleri = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    durma_esigi = 2.3
    hareket_durdu = False
    vurus_anı_bulundu = False
    vurus_zamani = 0
    hamle_bitis_zamani = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

         # HSV maskeleri oluşturma
        kirmizi_mask = cv2.inRange(hsv, kirmizi_alt, kirmizi_ust)
        beyaz_mask = cv2.inRange(hsv, beyaz_alt, beyaz_ust)
        
         # Erozyon ve dilatasyon işlemleri
        kirmizi_mask = cv2.erode(kirmizi_mask, None, iterations=2)
        kirmizi_mask = cv2.dilate(kirmizi_mask, None, iterations=2)
        beyaz_mask = cv2.erode(beyaz_mask, None, iterations=2)
        beyaz_mask = cv2.dilate(beyaz_mask, None, iterations=2)

        kirmizi_konturlar, _ = cv2.findContours(kirmizi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        beyaz_konturlar, _ = cv2.findContours(beyaz_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kirmizi_merkez = None

        if kirmizi_konturlar:
            en_buyuk = max(kirmizi_konturlar, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(en_buyuk)
            kirmizi_merkez = (int(x + w / 2), int(y + h / 2))
            cv2.circle(frame, kirmizi_merkez, 10, (250, 0, 20), 3)
            hareket_izleri.append(kirmizi_merkez)

            # Hareket izlerini çiz
            if len(hareket_izleri) > 1:
                for i in range(1, len(hareket_izleri)):
                    cv2.line(frame, hareket_izleri[i - 1], hareket_izleri[i], (0, 0, 255), 2)


        if onceki_kirmizi_merkez and kirmizi_merkez:
            mesafe = np.linalg.norm(np.array(kirmizi_merkez) - np.array(onceki_kirmizi_merkez))
            hiz = mesafe * fps / 1000  # Piksel/saniye olarak hız hesaplama
            if mesafe < durma_esigi:  # Eğer mesafe çok küçükse, top hareketsiz kabul edilebilir
                hiz = 0
                if not hareket_durdu:
                    hamle_bitis_zamani = current_time
                    hareket_durdu = True
            else:
                hareket_durdu = False

            if current_time - hamle_bitis_zamani < 3 and hareket_durdu:
                cv2.putText(frame, "Toplar durdu", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Hiz: {hiz:.2f} ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if beyaz_konturlar and not vurus_anı_bulundu:
            for kontur in beyaz_konturlar:
                x, y, w, h = cv2.boundingRect(kontur)
                beyaz_merkez = (int(x + w / 2), int(y + h / 2))
                if np.linalg.norm(np.array(beyaz_merkez) - np.array(kirmizi_merkez)) < 50:
                    vurus_anı_bulundu = True
                    vurus_zamani = current_time
                    cv2.putText(frame, "Hamle baslangici: Beyaz topa vuruldu", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if current_time - vurus_zamani < 3 and vurus_anı_bulundu:
            cv2.putText(frame, "Hamle baslangici: Beyaz topa vuruldu", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, "Q'ya basarak cikabilirsiniz", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        onceki_kirmizi_merkez = kirmizi_merkez
        cv2.imshow("Karambol Bilardo Top Takibi", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main('video1.avi')
