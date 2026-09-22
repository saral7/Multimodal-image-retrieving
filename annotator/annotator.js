var lista = data


const reference = document.getElementById("reference")
reference.style.width = "40vw"
reference.style.display = "flex"

const ref_img = document.getElementById("img-ref")
const tar_img = document.getElementById("img-tar")
const save_button = document.getElementById("save-button")
const download_button = document.getElementById("download-button")
const description = document.getElementById("description")
const relative_caption = document.getElementById("relative-caption")


save_button.addEventListener('click', function() {
   const obj = {
      "reference_img" : ref_img.src,
      "target_img" : tar_img.src,
      "description" : description.value,
      "relative_caption" : relative_caption.value
   }
   
   description.value = ""
   relative_caption.value = ""

   fetch('http://localhost:3000/write', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify({ obj }),
   })
   .catch(error => {
         console.error('Error:', error);
   });
});


for (img of lista.images) {
   var img_elem = document.createElement("img")
   img_elem.src = img.id
   img_elem.style.height = "100px"
   img_elem.style.border = "1px solid transparent"
   img_elem.addEventListener('click', function() {
      ref_img.src = this.src;
   });
   reference.appendChild(img_elem)
}

const target = document.getElementById("target")
target.style.width = "40vw"
target.style.display = "flex"

for (img of lista.images) {
   var img_elem = document.createElement("img")
   img_elem.src = img.id
   img_elem.style.height = "100px"
   img_elem.addEventListener('click', function() {
      tar_img.src = this.src;
   });
   target.appendChild(img_elem)
}


