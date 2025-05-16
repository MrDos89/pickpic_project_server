package com.febrie.store;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiConsumer;

@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class DataStore {
    private final Map<String, String> store = new ConcurrentHashMap<>();
    
    private static DataStore instance;
    
    public static synchronized DataStore getInstance() {
        if (instance == null) {
            instance = new DataStore();
        }
        return instance;
    }
    
    public void put(String key, String value) {
        store.put(key, value);
    }
    
    public String get(String key) {
        return store.get(key);
    }
    
    public String remove(String key) {
        return store.remove(key);
    }
    
    public boolean containsKey(String key) {
        return store.containsKey(key);
    }
    
    public void clear() {
        store.clear();
    }
    
    public int size() {
        return store.size();
    }
    
    public boolean isEmpty() {
        return store.isEmpty();
    }
    
    public void forEach(BiConsumer<String, String> action) {
        store.forEach(action);
    }
    
    public Map<String, String> getAll() {
        return Map.copyOf(store);
    }
}
