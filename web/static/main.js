function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#image_upload_preview').attr('src', e.target.result);
        }

        reader.readAsDataURL(input.files[0]);
    }
}


$(document).ready(function(){
    $("#inputImage").change(function () {
        readURL(this);
    });

    $("#inputForm").submit(function (e) {
        e.preventDefault();

        let form = $('#inputForm')[0];
        var fd = new FormData(form);

        fetch('/getcaption', {method: 'POST', body: fd})
            .then(res => {
                console.log("res: ", res);
                return res.json() // or res.text, res.arraybuffer
            })
            .then(result => {
                console.log("result: ", result);

                $("#image-caption-text").text(result.text);
            })

        $("#image-caption-block").show();
    });
});