package com.febrie.util;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;

public class FileManager {

    public static void saveBase64ImageToFile(String base64Image, String fileName) throws IOException {
        String directory = Config.getImageSavePath();
        Path dirPath = Paths.get(directory);

        if (!Files.exists(dirPath)) {
            Files.createDirectories(dirPath);
            Logger.info("이미지 저장 디렉토리 생성: " + directory);
        }

        if (!fileName.toLowerCase().endsWith(".jpg")) {
            fileName = fileName + ".jpg";
        }

        String filePath = directory + File.separator + fileName;
        File file = new File(filePath);

        if (file.exists()) {
            file.delete();
        }

        if (!file.exists()) {
            if (!file.getParentFile().exists())
                file.getParentFile().mkdirs();
        }

        try {
            byte[] imageData = Base64.getDecoder().decode(base64Image);

            try (FileOutputStream fos = new FileOutputStream(filePath)) {
                fos.write(imageData);
            }

            Logger.success("이미지 파일 저장 완료: " + filePath);
        } catch (IllegalArgumentException e) {
            Logger.error("잘못된 Base64 형식: " + e.getMessage());
            throw e;
        } catch (IOException e) {
            Logger.error("파일 저장 오류: " + e.getMessage());
            throw e;
        }
    }
}
