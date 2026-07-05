let char_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';

let title = 'BlackAngelSK.github.io';
let i = 0;
let direction = 1;

setInterval(() => {
    document.title = title.substring(0, i + 1);
    if (i === 0) {
        direction = 1;
    } else if (i === title.length) {
        direction = -1;
    }
    i += direction;
}, 400);