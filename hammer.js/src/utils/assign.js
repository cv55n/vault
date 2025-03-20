/**
 * @private
 * 
 * significa que as propriedades no dest serão reescritas por outras no src
 * 
 * @param {Object} target
 * @param {...Object} objects_to_assign
 * 
 * @returns {Object} alvo
 */
let assign;

if (typeof Object.assign !== 'function') {
    assign = function assign(target) {
        if (target === undefined || target === null) {
            throw new TypeError('não foi possível converter algo indefinido ou nulo em um objeto');
        }

        let output = Object(target);

        for (let index = 1; index < arguments.length; index++) {
            const source = arguments[index];

            if (source !== undefined && source !== null) {
                for (const nextKey in source) {
                    if (source.hasOwnProperty(nextKey)) {
                        output[nextKey] = source[nextKey];
                    }
                }
            }
        }

        return output;
    };
} else {
    assign = Object.assign;
}

export default assign;