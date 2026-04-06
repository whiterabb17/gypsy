use ollama_rs::Ollama;

fn main() {
    println!("Testing Ollama::new...");
    let host = "localhost".to_string();
    let port = 11434;
    let _ = Ollama::new(host, port);
    println!("Ollama::new(localhost) success!");

    let host = "127.0.0.1".to_string();
    let _ = Ollama::new(host, port);
    println!("Ollama::new(127.0.0.1) success!");
}
