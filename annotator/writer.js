const express = require('express');
const fs = require('fs');
const path = require('path');
var cors = require('cors')

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

app.use(cors())
app.use(express.static(__dirname));

app.post('/write', (req, res) => {
    console.log(req.body.obj)
    const dataToWrite = req.body.obj; 

    fs.readFile('annotations.json', 'utf8', (err, data) => {
        if (err) {
            console.error('Error reading file:', err);
            return; 
        }

        try {
            data = JSON.parse(data);
            
            data.images.push(dataToWrite);

            fs.writeFile('annotations.json', JSON.stringify(data, null, 2), (err) => {
                if (err) {
                    console.error('Error writing to file:', err);
                    return;
                }
                res.send('Data written successfully!');
            });
        } catch (parseError) {
            console.error('Error parsing JSON:', parseError);
        }
    });
});

app.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});