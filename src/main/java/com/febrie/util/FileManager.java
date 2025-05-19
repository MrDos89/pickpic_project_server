package com.febrie.util;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class FileManager {

    public static void saveBase64ImageToFile(String base64Image, String ssid, String uid) throws IOException {
        File file = new File(Config.getImageSavePath() + ssid + "/" + uid + ".jpg");
        try {
            byte[] imageData = Base64.getDecoder().decode(base64Image);
            try (FileOutputStream fos = new FileOutputStream(file)) {
                fos.write(imageData);
            }
            Logger.success("이미지 파일 저장 완료: " + file);
        } catch (IllegalArgumentException e) {
            Logger.error("잘못된 Base64 형식: " + e);
            throw e;
        } catch (IOException e) {
            Logger.error("파일 저장 오류: " + e);
            throw e;
        }
    }
}
