void setup() {
    // シリアル通信の開始
    Serial.begin(9600);
    
    // ピンのモードを設定
    pinMode(7, OUTPUT);
    pinMode(6, OUTPUT);
    digitalWrite(7, LOW);
    digitalWrite(6, LOW);
}

void loop() {
    // シリアルデータが受信された場合
    if (Serial.available() > 0) {
        char receivedChar = Serial.read(); 
        // 受信したデータに応じてピンを制御
        if (receivedChar == '1') {
            digitalWrite(7, HIGH);  
            delay(2000);
            digitalWrite(7, LOW);   
        } else if (receivedChar == '0') {
            digitalWrite(6, HIGH); 
            delay(2000);  
            digitalWrite(6, LOW);     
        }
    }
}
