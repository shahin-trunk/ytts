charset_hi = [' ', '!', '"', ',', '.', ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
              'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z', 'æ', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɒ', 'ɔ', 'ɖ', 'ə', 'ɚ',
              'ɛ', 'ɜ', 'ɟ', 'ɡ', 'ɣ', 'ɪ', 'ɲ', 'ɳ', 'ɹ', 'ɾ', 'ʂ', 'ʃ', 'ʈ', 'ʊ', 'ʋ', 'ʌ', 'ʒ', 'ʔ', 'ʰ', 'ˈ', 'ˌ',
              'ː', '̃', '̩', 'θ', 'χ', 'ᵻ']
charset_ar_gen = [' ', '!', '-', '.', ':', 'a', 'b', 'd', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'q', 'r', 's', 't',
                  'u', 'w', 'z', 'ð', 'ħ', 'ɡ', 'ɣ', 'ɹ', 'ʃ', 'ʒ', 'ʔ', 'ʕ', 'ˈ', 'ˌ', 'ː', 'ˤ', '̪', 'θ', 'χ', '،',
                  '؛', '؟']

charset_ar_org = [' ', '.', 'a', 'b', 'd', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'q', 'r', 's', 't', 'u', 'w', 'z',
                  'ð', 'ħ', 'ɡ', 'ɣ', 'ɹ', 'ʃ', 'ʒ', 'ʔ', 'ʕ', 'ˈ', 'ˌ', 'ː', 'ˤ', '̪', 'θ', 'χ']

charset_en_org = [' ', '!', '"', "'", ',', '-', '.', ':', ';', '?', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l',
                  'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z', 'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə',
                  'ɚ', 'ɛ', 'ɜ', 'ɡ', 'ɪ', 'ɬ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʔ', 'ʲ', 'ˈ', 'ˌ', 'ː', '̃', '̩', 'θ',
                  'ᵻ']

charset_en_gen = [' ', '!', '"', "'", ',', '.', ':', ';', '?', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm',
                  'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'z', 'æ', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ', 'ɛ',
                  'ɜ', 'ɡ', 'ɪ', 'ɬ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ', 'ʒ', 'ʔ', 'ˈ', 'ˌ', 'ː', '̃', '̩', 'θ', 'ᵻ']

charset_ml_org = [' ', 'a', 'b', 'c', 'd', 'e', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                  'p', 'r', 's', 't', 'u', 'v', 'z', 'ŋ', 'ɐ', 'ɑ', 'ɒ', 'ɔ', 'ɕ', 'ɖ', 'ə', 'ɛ', 'ɜ', 'ɟ', 'ɡ', 'ɨ',
                  'ɪ', 'ɭ', 'ɲ', 'ɳ', 'ɹ', 'ɾ', 'ʂ', 'ʃ', 'ʈ', 'ʊ', 'ʌ', 'ʒ', 'ʰ', 'ʲ', 'ˈ', 'ˌ', 'ː', '̩', 'θ']

charset_ml_org2 = [' ', 'a', 'b', 'c', 'd', 'e', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                   'p', 'r', 's', 't', 'u', 'v', 'ŋ', 'ɐ', 'ɕ', 'ɖ', 'ə', 'ɟ', 'ɡ', 'ɨ', 'ɪ', 'ɭ', 'ɲ', 'ɳ', 'ɾ', 'ʂ',
                   'ʈ', 'ʊ', 'ʰ', 'ʲ', 'ˈ', 'ˌ', 'ː', '̩']

PUNCTUATIONS_EN = ['!', '?', '-', ',', ';', ':', '.', '\'', '"']
PUNCTUATIONS_AR = ['!', '؟', '-', '،', '؛', ':', '.']
PUNCTUATIONS_HI = ['!', '?', '-', ',', ';', ':', '.', '\'', '"']
PUNCTUATIONS_ML = ['"', ',', '-', '.', ':', '?']
PUNCTUATIONS_ML2 = ['!', ',', '.', ':', ';', '?']

punctuation_set = set(PUNCTUATIONS_EN + PUNCTUATIONS_AR + PUNCTUATIONS_HI + PUNCTUATIONS_ML + PUNCTUATIONS_ML2)

charset = set()
for ch_set in [charset_hi, charset_en_gen, charset_en_org, charset_ar_org, charset_ar_gen, charset_ml_org,
               charset_ml_org2]:
    for ch in ch_set:
        charset.add(ch)

charset = set(sorted(charset))
punctuation_set = set(sorted(punctuation_set))

charset = sorted(set(sorted(charset - punctuation_set)))
punctuation_set = sorted(punctuation_set)

print(charset)
print(len(charset))

print(punctuation_set)
print(len(punctuation_set))
