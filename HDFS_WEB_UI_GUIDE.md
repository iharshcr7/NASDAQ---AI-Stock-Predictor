# 🌐 HDFS Web UI Verification Guide

## 📍 HDFS Web UI Location

**URL:** http://localhost:9870/explorer.html#/stock_data/live_api_dumps

This is where your CSV files from live predictions are stored in HDFS.

---

## ✅ Quick Verification Steps

### 1. Check HDFS is Running
```bash
jps
```
**Expected output:**
```
12345 NameNode
12346 DataNode
12347 SecondaryNameNode
```

If not running:
```bash
start-dfs.sh
```

---

### 2. Create HDFS Directory (if needed)
```bash
hdfs dfs -mkdir -p /stock_data/live_api_dumps
```

---

### 3. Run Verification Script
```bash
python verify_hdfs_upload.py
```

This script will:
- ✅ Check if HDFS is running
- ✅ Create the directory if needed
- ✅ List existing files
- ✅ Test upload functionality
- ✅ Show web UI URL

---

### 4. Run a Prediction
```bash
streamlit run app.py
```

Then:
1. Select a stock (e.g., AAPL)
2. Click "Predict Live"
3. Wait for completion
4. Check dashboard status (should show "✅ Uploaded")

---

### 5. View Files in Web UI

**Open in browser:**
```
http://localhost:9870/explorer.html#/stock_data/live_api_dumps
```

**You should see:**
- CSV files with format: `{SYMBOL}_live_{TIMESTAMP}.csv`
- Example: `AAPL_live_2026_04_29_143052.csv`
- File size, modification time, permissions

---

## 🔍 Manual Verification Commands

### List Files in HDFS
```bash
hdfs dfs -ls /stock_data/live_api_dumps/
```

**Expected output:**
```
Found 3 items
-rw-r--r--   1 user supergroup   1234 2026-04-29 14:30 /stock_data/live_api_dumps/AAPL_live_2026_04_29_143052.csv
-rw-r--r--   1 user supergroup   1234 2026-04-29 14:35 /stock_data/live_api_dumps/MSFT_live_2026_04_29_143512.csv
-rw-r--r--   1 user supergroup   1234 2026-04-29 14:40 /stock_data/live_api_dumps/GOOGL_live_2026_04_29_144023.csv
```

---

### View File Content
```bash
hdfs dfs -cat /stock_data/live_api_dumps/AAPL_live_2026_04_29_143052.csv | head -5
```

---

### Check File Size
```bash
hdfs dfs -du -h /stock_data/live_api_dumps/
```

---

### Count Files
```bash
hdfs dfs -count /stock_data/live_api_dumps/
```

---

## 🌐 HDFS Web UI Features

### Main Dashboard
**URL:** http://localhost:9870

**Shows:**
- Cluster overview
- Node status
- Storage capacity
- Live nodes

### File Browser
**URL:** http://localhost:9870/explorer.html

**Features:**
- Browse HDFS directories
- View file details
- Download files
- Check permissions
- See file sizes

### Navigate to Your Files
1. Open: http://localhost:9870/explorer.html
2. Click on `/` (root)
3. Click on `stock_data/`
4. Click on `live_api_dumps/`
5. See your CSV files!

**Direct link:**
http://localhost:9870/explorer.html#/stock_data/live_api_dumps

---

## 📊 What You Should See

### After First Prediction
```
/stock_data/live_api_dumps/
└── AAPL_live_2026_04_29_143052.csv
```

### After Multiple Predictions
```
/stock_data/live_api_dumps/
├── AAPL_live_2026_04_29_143052.csv
├── AAPL_live_2026_04_29_150123.csv
├── MSFT_live_2026_04_29_143512.csv
├── GOOGL_live_2026_04_29_144023.csv
├── AMZN_live_2026_04_29_145234.csv
└── NVDA_live_2026_04_29_151045.csv
```

---

## 🎯 Dashboard Integration

### In Streamlit Dashboard

After clicking "Predict Live", you'll see:

**Pipeline Status:**
```
┌─────────────────────────────────────┐
│ CSV Dump        │ ✅ Saved          │
│ HDFS Upload     │ ✅ Uploaded       │
│ MongoDB         │ ✅ Saved          │
└─────────────────────────────────────┘
```

**Storage Information (Expandable):**
```
Local Storage:
  ✅ CSV saved: AAPL_live_2026_04_29_143052.csv
  Full path: D:\Stock predictor\live_api_dumps\AAPL_live_2026_04_29_143052.csv
  File size: 1,234 bytes

HDFS Storage:
  ✅ Uploaded to HDFS
  Path: hdfs://localhost:9000/stock_data/live_api_dumps/AAPL_live_2026_04_29_143052.csv
  Verify: hdfs dfs -ls /stock_data/live_api_dumps/
```

---

## 🔧 Troubleshooting

### Issue 1: HDFS Not Running
**Symptom:** Web UI not accessible at http://localhost:9870

**Solution:**
```bash
# Check status
jps

# Start HDFS
start-dfs.sh

# Wait 30 seconds, then check again
jps
```

---

### Issue 2: Directory Not Found
**Symptom:** 404 error in web UI

**Solution:**
```bash
# Create directory
hdfs dfs -mkdir -p /stock_data/live_api_dumps

# Verify
hdfs dfs -ls /stock_data/
```

---

### Issue 3: No Files Showing
**Symptom:** Directory exists but empty

**Solution:**
1. Run a prediction from dashboard
2. Check dashboard shows "✅ Uploaded"
3. Refresh web UI
4. Verify with command:
   ```bash
   hdfs dfs -ls /stock_data/live_api_dumps/
   ```

---

### Issue 4: Upload Failed
**Symptom:** Dashboard shows "⚠️ Failed"

**Solution:**
```bash
# Check HDFS is running
jps

# Check directory exists
hdfs dfs -ls /stock_data/

# Check permissions
hdfs dfs -ls -d /stock_data/live_api_dumps/

# Try manual upload
hdfs dfs -put test.csv /stock_data/live_api_dumps/
```

---

### Issue 5: Web UI Shows Wrong Directory
**Symptom:** Files not in expected location

**Solution:**
1. Check HDFS configuration in `scripts/predict_live.py`:
   ```python
   HDFS_LIVE_DUMPS_PATH = "/stock_data/live_api_dumps/"
   ```
2. Search for files:
   ```bash
   hdfs dfs -find / -name "*_live_*.csv"
   ```

---

## 📋 Verification Checklist

Before demo, verify:

- [ ] HDFS is running (`jps` shows NameNode, DataNode)
- [ ] Web UI accessible at http://localhost:9870
- [ ] Directory exists: `/stock_data/live_api_dumps/`
- [ ] Can browse to directory in web UI
- [ ] Run test prediction
- [ ] CSV appears in web UI
- [ ] Dashboard shows "✅ Uploaded"
- [ ] Can view file details in web UI

---

## 🎬 Demo Flow with Web UI

### During Demo:

1. **Show Web UI Before Prediction**
   - Open: http://localhost:9870/explorer.html#/stock_data/live_api_dumps
   - Show current files (or empty directory)
   - Note the count

2. **Run Prediction**
   - `streamlit run app.py`
   - Select AAPL
   - Click "Predict Live"
   - Show progress bar

3. **Show Dashboard Status**
   - Point to "✅ Uploaded" status
   - Expand "Storage Information"
   - Show HDFS path

4. **Refresh Web UI**
   - Go back to web UI
   - Refresh page (F5)
   - Show new CSV file appeared
   - Click on file to show details

5. **Verify with Command**
   ```bash
   hdfs dfs -ls /stock_data/live_api_dumps/
   ```
   - Show file in terminal
   - Matches web UI

**This proves real HDFS integration!**

---

## 🌟 Key Points for Viva

### Question: "How do you know files are really in HDFS?"

**Answer:**
> "We can verify in three ways:
> 
> 1. **Dashboard Status:** Shows '✅ Uploaded' after prediction
> 
> 2. **HDFS Web UI:** Open http://localhost:9870/explorer.html#/stock_data/live_api_dumps and see the CSV files with timestamps
> 
> 3. **Command Line:** Run `hdfs dfs -ls /stock_data/live_api_dumps/` and see the files listed
> 
> All three methods show the same files, proving real HDFS integration."

**Demo:** Show all three methods during presentation

---

## 📞 Quick Reference

### URLs
- **Main Dashboard:** http://localhost:9870
- **File Browser:** http://localhost:9870/explorer.html
- **Your Files:** http://localhost:9870/explorer.html#/stock_data/live_api_dumps

### Commands
```bash
# Check HDFS
jps

# List files
hdfs dfs -ls /stock_data/live_api_dumps/

# Count files
hdfs dfs -count /stock_data/live_api_dumps/

# View file
hdfs dfs -cat /stock_data/live_api_dumps/AAPL_live_*.csv | head
```

### Verification Script
```bash
python verify_hdfs_upload.py
```

---

## ✅ Success Indicators

You know HDFS upload is working when:

1. ✅ Dashboard shows "✅ Uploaded"
2. ✅ Web UI shows CSV files at http://localhost:9870/explorer.html#/stock_data/live_api_dumps
3. ✅ Command `hdfs dfs -ls /stock_data/live_api_dumps/` lists files
4. ✅ File timestamps match prediction time
5. ✅ File names follow pattern: `{SYMBOL}_live_{TIMESTAMP}.csv`
6. ✅ File sizes are reasonable (1-5 KB)

---

**Your HDFS integration is production-ready! 🚀**

Use the web UI to demonstrate real distributed storage during your demo.
