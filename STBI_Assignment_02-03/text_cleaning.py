import pandas as pd
import re
import pickle
import time
import multiprocessing as mp
from functools import partial
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import psutil
import threading
from tqdm import tqdm

class MaxCPUPreprocessor:
    def __init__(self):
        print("Initializing MAX CPU preprocessor...")
        self.stopwords_id = set(stopwords.words('indonesian'))
        self.clean_pattern = re.compile(r'[^\w\s]')
        print("Preprocessor ready for maximum CPU utilization!")
    
    def worker_init(self):
        """Initialize worker process dengan stemmer lokal"""
        # Setiap worker punya stemmer sendiri (menghindari pickling issues)
        import nltk
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        
        global worker_stemmer, worker_stopwords, worker_pattern
        worker_stemmer = StemmerFactory().create_stemmer()
        worker_stopwords = set(stopwords.words('indonesian'))
        worker_pattern = re.compile(r'[^\w\s]')
    
    def preprocess_single_worker(self, text):
        """Worker function - optimized untuk speed"""
        if not text or pd.isna(text):
            return ""
        
        # Fast string operations
        text = str(text).lower()
        text = worker_pattern.sub('', text)
        
        # Optimized tokenization
        tokens = [word for word in text.split() 
                 if word and len(word) > 2 and word not in worker_stopwords]
        
        if not tokens:
            return ""
        
        # Stem everything in one go
        clean_text = " ".join(tokens)
        return worker_stemmer.stem(clean_text)
    
    def cpu_monitor(self, stop_event, results_queue):
        """Monitor CPU usage secara real-time"""
        max_cpu = 0
        cpu_history = []
        
        while not stop_event.is_set():
            cpu_percent = psutil.cpu_percent(interval=0.5, percpu=True)
            avg_cpu = sum(cpu_percent) / len(cpu_percent)
            max_cpu = max(max_cpu, avg_cpu)
            cpu_history.append(avg_cpu)
            
            if len(cpu_history) > 10:
                cpu_history.pop(0)
            
            time.sleep(0.5)
        
        results_queue.put({
            'max_cpu': max_cpu,
            'avg_cpu': sum(cpu_history) / len(cpu_history) if cpu_history else 0
        })

    def preprocess_max_cpu(self, texts, n_processes=None):
        """Maximum CPU utilization preprocessing"""
        
        if n_processes is None:
            n_processes = mp.cpu_count()  # semua logical cores
        
        print(f"\n{'='*70}")
        print(f"MAXIMUM CPU UTILIZATION MODE")
        print(f"Dataset: {len(texts)} documents")
        print(f"Processes: {n_processes} (using ALL logical cores)")
        print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        print(f"{'='*70}")
        
        # Setup CPU monitoring
        stop_monitor = threading.Event()
        monitor_queue = mp.Queue()
        monitor_thread = threading.Thread(
            target=self.cpu_monitor, 
            args=(stop_monitor, monitor_queue)
        )
        monitor_thread.start()
        
        start_time = time.time()
        
        with mp.Pool(
            processes=n_processes,
            initializer=self.worker_init,
            maxtasksperchild=1000
        ) as pool:
            
            chunk_size = max(1, len(texts) // (n_processes * 4))
            batch_size = n_processes * 100
            
            results = []
            processed = 0
            
            # >>> pakai tqdm di sini
            for i in tqdm(range(0, len(texts), batch_size), desc="Preprocessing", unit="docs"):
                batch_start = time.time()
                batch = texts[i:i+batch_size]
                
                batch_results = pool.map(
                    self.preprocess_single_worker,
                    batch,
                    chunksize=chunk_size
                )
                
                results.extend(batch_results)
                processed += len(batch)
                
                batch_time = time.time() - batch_start
                speed = len(batch) / batch_time
                
                elapsed = time.time() - start_time
                if processed > 0:
                    eta = (elapsed / processed) * (len(texts) - processed)
                    eta_min = int(eta // 60)
                    eta_sec = int(eta % 60)
                else:
                    eta_min = eta_sec = 0
                
                print(f" Batch done | Speed: {speed:6.1f} docs/s | "
                    f"ETA: {eta_min:02d}:{eta_sec:02d} | "
                    f"Processed: {processed}/{len(texts)}")
        
        stop_monitor.set()
        monitor_thread.join()
        
        try:
            cpu_stats = monitor_queue.get_nowait()
        except:
            cpu_stats = {'max_cpu': 0, 'avg_cpu': 0}
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average speed: {len(texts)/total_time:.1f} docs/second")
        print(f"Peak CPU usage: {cpu_stats['max_cpu']:.1f}%")
        print(f"Average CPU usage: {cpu_stats['avg_cpu']:.1f}%")
        print(f"CPU efficiency: {cpu_stats['avg_cpu']/100*n_processes:.1f}/{n_processes} cores utilized")
        print(f"{'='*70}")
        
        return results
    
    def preprocess_ultra_fast(self, texts):
        """Ultra-fast version - minimal processing untuk maximum speed"""
        print("\nULTRA-FAST MODE (minimal stemming)")
        
        def ultra_fast_worker(text):
            if not text or pd.isna(text):
                return ""
            
            # Minimal processing
            text = str(text).lower()
            words = text.split()
            
            # Quick filtering
            filtered = [w for w in words if len(w) > 3 and w.isalpha()]
            
            return " ".join(filtered)
        
        start_time = time.time()
        
        # Use ALL cores
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(ultra_fast_worker, texts, chunksize=50)
        
        end_time = time.time()
        print(f"Ultra-fast processing: {end_time - start_time:.2f}s")
        print(f"Speed: {len(texts)/(end_time - start_time):.1f} docs/second")
        
        return results

def benchmark_comparison(texts_sample):
    """Benchmark different approaches"""
    print(f"\n{'='*50}")
    print("BENCHMARKING DIFFERENT APPROACHES")
    print(f"Sample size: {len(texts_sample)} documents")
    print(f"{'='*50}")
    
    preprocessor = MaxCPUPreprocessor()
    
    # Test 1: Standard approach
    print("\n1. Standard approach (6 processes)...")
    start = time.time()
    # results1 = preprocessor.preprocess_max_cpu(texts_sample[:100], n_processes=6)
    time1 = time.time() - start
    
    # Test 2: Max CPU approach
    print("\n2. Max CPU approach (12 processes)...")
    start = time.time()
    results2 = preprocessor.preprocess_max_cpu(texts_sample[:100], n_processes=12)
    time2 = time.time() - start
    
    # Test 3: Ultra fast approach
    print("\n3. Ultra-fast approach...")
    start = time.time()
    results3 = preprocessor.preprocess_ultra_fast(texts_sample[:100])
    time3 = time.time() - start
    
    print(f"\n{'='*50}")
    print("BENCHMARK RESULTS:")
    # print(f"Standard (6 proc):  {time1:.2f}s")
    print(f"Max CPU (12 proc):  {time2:.2f}s")
    print(f"Ultra-fast:         {time3:.2f}s")
    print(f"{'='*50}")

def main():
    """Main function dengan options"""
    
    preprocessor = MaxCPUPreprocessor()
    
    # Load data
    print("Loading dataset...")
    try:
        df = pd.read_csv("News.csv")
        contents = df['content'].fillna('').tolist()
        print(f"✓ Loaded {len(contents)} documents")
    except FileNotFoundError:
        print("❌ News.csv not found!")
        return
    
    # Show options
    print(f"\nProcessing options:")
    print("1. Maximum CPU utilization (recommended for 14K docs)")
    print("2. Ultra-fast mode (no stemming, fastest)")
    print("3. Benchmark comparison (small sample)")
    
    choice = input("\nChoose option (1-3, default=1): ").strip()
    
    if choice == "3":
        benchmark_comparison(contents)
        return
    elif choice == "2":
        results = preprocessor.preprocess_ultra_fast(contents)
    else:
        # Default: Maximum CPU
        results = preprocessor.preprocess_max_cpu(contents, n_processes=12)
    
    # Save results
    df['content_clean'] = results
    df.to_csv('News_processed.csv', index=False)
    
    # Stats
    non_empty = [r for r in results if r.strip()]
    print(f"\nFINAL STATS:")
    print(f"Success rate: {len(non_empty)/len(results)*100:.1f}%")
    print(f"Results saved to: News_processed.csv")

if __name__ == "__main__":
    main()