from model import ModelWrapper
model = ModelWrapper.get_shared()

## your telegram bot here

## smth like getClassificationResult method, you can use your own,
# it's just like an example: 
@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    try:
        # dont know maybe it doesnt work
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        src = 'C:\\target_path_to_file' + file_info.file_path[7:]
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        result_classification = model.predict(src)    ## result string such as sunflower or dandelion 

    except (AttributeError, cv2.error, IOError, ImportError, IndexError, KeyError, NameError, OSError, \
        SyntaxError, TypeError, ValueError, IndexError, ZeroDivisionError, RuntimeError):
        bot.reply_to(message, CANT_FIND)
    
