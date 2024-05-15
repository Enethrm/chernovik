import requests

from aiogram import Router, F, types, Bot
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from helpers.text_helper import get_text_from_config


class ColorizationStates(StatesGroup):
    start_colorization = State()
    send_pic = State()


colorization_router = Router()


@colorization_router.message(F.text.lower() == 'колоризация')
async def colorization_get_photo(message: types.Message, state: FSMContext):
    await message.answer(get_text_from_config('post_photo', block='RESPONSE'))
    await state.set_state(ColorizationStates.start_colorization)
    

@colorization_router.message(ColorizationStates.start_colorization, F.photo)
async def colorization_send_photo(message: types.Message, state: FSMContext, bot: Bot):
    path_to_photo = f'./temp/{message.photo[-1].file_unique_id}.jpg'
    await bot.download(message.photo[-1], destination=path_to_photo)
    await message.answer(get_text_from_config('api_proccess', block='RESPONSE'))

    proccess_file = requests.post('http://api:8000/colorization/', files={'file': open(path_to_photo, 'rb')})
    with open(path_to_photo, 'wb') as photo:
        photo.write(proccess_file.content)
    await message.answer_photo(photo=types.FSInputFile(path=path_to_photo))