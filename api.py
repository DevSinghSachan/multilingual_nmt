
from nfseq import translate

sent = '道元（どうげん）は、鎌倉時代初期の禅僧。'
sent_list = translate(sent=sent, src='ja', tgt='en')
print(sent_list)
