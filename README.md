potrebujete python knihovnu ultralytics
training data jsou v zipu, ten rozbalte, tak aby slozky train,val byli v adresari tam kde mas skripty
zpust process_labels.py, kterej upravi labels
a pak jenom spustit train.py

data a modely jsou v: https://drive.google.com/drive/folders/1zfFXMRVMnHY0xS7eI3EbHi3oTY3fBAH_?usp=drive_link

## Spustenie trackovania:

### Základné spustenie videa:
```bash
python3 track.py --input video.mp4 --model best.pt
```

### Parametre:
- `--conf-thresh 0.x` - citlivosť detekcie (vyššie = menej detekcií, rýchlejšie)

### Ovládanie:
- **Q** - ukončiť
- **P** - pauza/pokračovať
