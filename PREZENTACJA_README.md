# Prezentacja Systemu Rekomendacji Filmów - Instrukcja

Ten folder zawiera materiały do prezentacji projektu systemu rekomendacji filmów. Prezentacja została przygotowana w języku polskim i jest przeznaczona na 30-minutowe wystąpienie. Prezentacja zawiera lekkie tło oraz subtelne elementy stylistyczne, które nadają jej profesjonalny wygląd.

## Zawartość

1. `Prezentacja_Systemu_Rekomendacji_Filmow.md` - plik Markdown zawierający treść prezentacji
2. `markdown_to_pptx.py` - skrypt Python do konwersji pliku Markdown na prezentację PowerPoint z dodatkowym formatowaniem i stylizacją

## Jak wygenerować prezentację PowerPoint

Aby wygenerować plik PowerPoint (.pptx) z pliku Markdown, należy:

1. Zainstalować wymagane biblioteki:
   ```
   pip install python-pptx
   ```

2. Uruchomić skrypt konwersji:
   ```
   python markdown_to_pptx.py
   ```

3. Skrypt wygeneruje plik `Prezentacja_Systemu_Rekomendacji_Filmow.pptx` w bieżącym katalogu.

## Struktura prezentacji

Prezentacja składa się z następujących sekcji:

1. Wprowadzenie do projektu
2. Zaawansowane metody dziedziczenia
3. Elementy uczenia maszynowego
4. Techniki rekomendacji stosowane przez Netflix i YouTube
5. Struktura projektu
6. Demonstracja systemu
7. Plany rozwoju
8. Podsumowanie

## Elementy stylistyczne prezentacji

Prezentacja zawiera następujące elementy stylistyczne:

1. **Specjalny slajd tytułowy** - pierwszy slajd ma specjalne formatowanie z większym tytułem i podtytułem
2. **Lekkie tło** - wszystkie slajdy mają delikatne jasnoniebieskie tło
3. **Spójne formatowanie tekstu** - jednolite czcionki i kolory dla wszystkich slajdów
4. **Wyróżnione bloki kodu** - kod jest formatowany z użyciem czcionki Courier New i niebieskiego koloru
5. **Slajd końcowy** - specjalny slajd "Dziękuję za uwagę" na zakończenie prezentacji
6. **Elementy dekoracyjne** - subtelne linie oddzielające na slajdzie tytułowym i końcowym

## Modyfikacja prezentacji

Jeśli chcesz zmodyfikować prezentację:

1. Edytuj plik `Prezentacja_Systemu_Rekomendacji_Filmow.md` według potrzeb
2. Pamiętaj, że separator `---` oznacza nowy slajd
3. Nagłówki (zaczynające się od `#`, `##` lub `###`) są używane jako tytuły slajdów
4. Bloki kodu (otoczone ```) są formatowane jako kod z odpowiednią czcionką
5. Po wprowadzeniu zmian, uruchom ponownie skrypt konwersji

## Dostosowanie stylu

Jeśli chcesz dostosować styl prezentacji:

1. Otwórz plik `markdown_to_pptx.py` i znajdź funkcje odpowiedzialne za stylizację:
   - `apply_slide_styling` - formatowanie slajdów
   - `apply_text_styling` - formatowanie tekstu
   - `create_title_slide` - formatowanie slajdu tytułowego
   - `create_closing_slide` - formatowanie slajdu końcowego

2. Możesz zmienić kolory tła, czcionek, marginesy i inne elementy stylistyczne

## Wskazówki do prezentacji

1. Prezentacja jest zaprojektowana na około 30 minut
2. Każdy slajd powinien zająć około 1-2 minuty
3. Slajdy z kodem mogą wymagać dłuższego omówienia
4. Warto przygotować dodatkowe przykłady lub demonstrację działania systemu

## Uwagi

- Skrypt konwersji tworzy prezentację z subtelnym formatowaniem, które nadaje jej profesjonalny wygląd
- Po wygenerowaniu pliku PowerPoint, możesz go dalej edytować bezpośrednio w programie PowerPoint
- Jeśli prezentacja będzie wyświetlana na różnych komputerach, upewnij się, że czcionka Courier New jest dostępna (używana do formatowania kodu)
