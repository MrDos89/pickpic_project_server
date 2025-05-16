package com.febrie.server;

import java.util.concurrent.atomic.AtomicLong;

public class Stats {
    private static final AtomicLong requestCount = new AtomicLong(0);
    
    private Stats() {}
    
    public static void incrementRequestCount() {
        requestCount.incrementAndGet();
    }
    
    public static long getRequestCount() {
        return requestCount.get();
    }
}