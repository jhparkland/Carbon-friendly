// assets/custom_script.js
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        open_new_tab: function(url) {
            if (url) {
                window.open(url, '_blank');
            }
            return window.dash_clientside.no_update;
        }
    }
});