from nfseq import translate

sent = '道元（どうげん）は、鎌倉時代初期の禅僧。'
sent = 'Examples of Koan stories in the beginning.'
print(sent)
sent_list = translate(sent=sent, src='en', tgt='ja')
print(sent_list)
sent_list = translate(sent=sent_list[0], src='ja', tgt='en')
print(sent_list)