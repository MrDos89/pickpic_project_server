package com.febrie.data;

import com.febrie.util.Config;

import java.io.File;
import java.util.Arrays;
import java.util.Objects;

public class DataManager {

    public static int getFolders() {
        File file = new File(Config.getImageSavePath());
        if (!file.exists()) return 0;
        File[] files = file.listFiles();
        return files == null ? 0 : files.length;
    }

    public static int getFiles() {
        if (getFolders() == 0) return 0;
        File[] files = new File(Config.getImageSavePath()).listFiles();
        if (files == null) return 0;
        return Arrays.stream(files).map(File::listFiles).filter(Objects::nonNull).mapToInt(temp -> temp.length).sum();
    }
}
